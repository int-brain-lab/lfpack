import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import lfpack


def _synthetic_lfp(nc=32, ns=500):
    """Return a deterministic low-rank (nc, ns) float32 array that resembles LFP structure."""
    t = np.linspace(0, 4 * np.pi, ns)
    temporal = np.array([np.sin(t), np.cos(t), np.sin(2 * t)])  # (3, ns)
    # spatial weights: channel index modulated by three patterns, scaled to LFP µV range
    ch = np.arange(nc, dtype=np.float64)
    spatial = np.column_stack(
        [
            np.sin(ch / nc * np.pi),
            np.cos(ch / nc * np.pi),
            np.sin(2 * ch / nc * np.pi),
        ]
    ) * np.array([100.0, 30.0, 10.0])  # (nc, 3)
    # fixed-seed noise from bit-reproducible hash, avoids BLAS subnormal FPU warnings
    noise = (np.arange(nc * ns, dtype=np.float32).reshape(nc, ns) % 17 - 8) * 0.1
    return (spatial @ temporal + noise).astype(np.float32)


class TestLfpack(unittest.TestCase):
    def setUp(self):
        self.data = _synthetic_lfp()
        self.nc, self.ns = self.data.shape

    def test_compress_output_shapes(self):
        c = lfpack.compress(self.data)
        r = c.U_scaled.shape[1]
        self.assertEqual(c.U_scaled.shape, (self.nc, r))
        self.assertEqual(c.Vh_hat.shape[0], r)
        # WP transform pads to leaf-node slots (>= ns_original)
        self.assertGreaterEqual(c.Vh_hat.shape[1], self.ns)
        self.assertEqual(c.ns_original, self.ns)

    def test_compress_ratios_positive(self):
        c = lfpack.compress(self.data)
        self.assertGreater(c.cr_svd, 1.0)
        self.assertGreater(c.cr_wp, 1.0)
        self.assertGreater(c.cr_total, 1.0)
        # cr_total = original_floats / stored_floats (U_scaled rows + non-zero Vh coefficients)
        r = c.U_scaled.shape[1]
        n_kept = int(np.count_nonzero(c.Vh_hat))
        expected_cr = float(self.nc * self.ns) / (r * self.nc + n_kept)
        self.assertAlmostEqual(c.cr_total, expected_cr, places=6)

    def test_svd_only_alpha_zero(self):
        c = lfpack.compress(self.data, alpha=0.0)
        self.assertEqual(c.cr_wp, 1.0)
        self.assertEqual(c.cr_total, c.cr_svd)

    def test_decompress_shape_and_dtype(self):
        c = lfpack.compress(self.data)
        rec = lfpack.decompress(c)
        self.assertEqual(rec.shape, self.data.shape)
        self.assertEqual(rec.dtype, np.float32)

    def test_decompress_fidelity(self):
        """Reconstruction RMSE should be well below signal RMS."""
        c = lfpack.compress(self.data)
        rec = lfpack.decompress(c)
        rms = float(np.sqrt(np.mean(self.data.astype(np.float64) ** 2)))
        rmse = float(np.sqrt(np.mean((self.data.astype(np.float64) - rec.astype(np.float64)) ** 2)))
        snr = 20.0 * np.log10(rms / max(rmse, 1e-12))
        self.assertGreater(snr, 10.0, f"SNR {snr:.1f} dB is too low")

    def test_svd_only_exact_reconstruction(self):
        """With alpha=0 and low epsilon on clean low-rank data, error should be tiny."""
        data = _synthetic_lfp()
        c = lfpack.compress(data, epsilon=1.0, alpha=0.0)
        rec = lfpack.decompress(c)
        rmse = float(np.sqrt(np.mean((data.astype(np.float64) - rec.astype(np.float64)) ** 2)))
        self.assertLess(rmse, 1.0)

    def test_higher_epsilon_lower_rank(self):
        """A larger epsilon should select fewer singular values (lower rank)."""
        r_low_eps = lfpack.compress(self.data, epsilon=10.0).U_scaled.shape[1]
        r_high_eps = lfpack.compress(self.data, epsilon=500.0).U_scaled.shape[1]
        self.assertGreaterEqual(r_low_eps, r_high_eps)

    def test_decompress_bin_channels_shape(self):
        """decompress(bin_channels=N) returns (nc//N, ns)."""
        c = lfpack.compress(self.data)
        for n in (2, 4):
            rec = lfpack.decompress(c, bin_channels=n)
            self.assertEqual(rec.shape, (self.nc // n, self.ns))
            self.assertEqual(rec.dtype, np.float32)

    def test_decompress_bin_channels_math(self):
        """Binned result equals sum of adjacent channels in the full reconstruction."""
        c = lfpack.compress(self.data, alpha=0.0)  # SVD-only for exact linearity
        full = lfpack.decompress(c)  # (nc, ns)
        for n in (2, 4):
            binned = lfpack.decompress(c, bin_channels=n)  # (nc//n, ns)
            nc_binned = self.nc // n
            expected = full[: nc_binned * n].reshape(nc_binned, n, self.ns).sum(axis=1)
            np.testing.assert_allclose(binned.astype(np.float64), expected.astype(np.float64), rtol=2e-5)


class TestLFPackH5(unittest.TestCase):
    """Round-trip tests for compress_to_h5 / LFPackReader using an in-memory dummy dataset."""

    NC = 32
    NS = 1000  # 4 chunks at CHUNK=256
    CHUNK = 256
    OVERLAP = 32

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        # compress_to_h5 expects (ns, nc) time-first float32
        data_nc_ns = _synthetic_lfp(nc=self.NC, ns=self.NS)
        self.npy = self.tmp_path / "checkpoint.npy"
        np.save(self.npy, data_nc_ns.T.astype(np.float32))  # (ns, nc)
        # minimal probe geometry — avoids requiring neuropixel in tests
        self.h = {
            "x": np.zeros(self.NC, dtype=np.float32),
            "y": np.arange(self.NC, dtype=np.float32) * 25.0,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, h5_path, recording, scale=0):
        lfpack.compress_to_h5(
            self.npy,
            h5_path,
            recording=recording,
            scale=scale,
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            n_jobs=1,
        )

    # ── single-recording round-trip ───────────────────────────────────────────

    def test_roundtrip_shape_and_dtype(self):
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        sr = lfpack.LFPackReader(h5)
        self.assertEqual(sr.nc, self.NC)
        self.assertEqual(sr.ns, self.NS)
        data, sync = sr.read_samples(0, self.NS)
        self.assertIsNone(sync)
        self.assertEqual(data.shape, (self.NS, self.NC))
        self.assertEqual(data.dtype, np.float32)

    def test_roundtrip_fidelity(self):
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        sr = lfpack.LFPackReader(h5)
        recon, _ = sr.read_samples(0, self.NS)
        original = np.load(self.npy).T  # back to (nc, ns)
        rms = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        rmse = float(np.sqrt(np.mean((original.astype(np.float64) - recon.T.astype(np.float64)) ** 2)))
        snr = 20.0 * np.log10(rms / max(rmse, 1e-12))
        self.assertGreater(snr, 5.0, f"SNR {snr:.1f} dB is too low")

    # ── recordings() / scales() catalogue ────────────────────────────────────

    def test_recordings_single(self):
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        self.assertEqual(lfpack.LFPackReader.recordings(h5), ["rec_a"])

    def test_scales_single(self):
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        self.assertEqual(lfpack.LFPackReader.scales(h5, "rec_a"), [0])

    def test_scales_two_levels(self):
        h5 = self.tmp_path / "pyramid.h5"
        # write two scale levels into the same file (append mode)
        self._write(h5, "rec_a", scale=0)
        with h5py.File(h5, "a") as dst:
            h5_s1 = self.tmp_path / "scale1.h5"
            self._write(h5_s1, "rec_a", scale=1)
            with h5py.File(h5_s1, "r") as src:
                src.copy("rec_a/01", dst["rec_a"])
        self.assertEqual(lfpack.LFPackReader.scales(h5, "rec_a"), [0, 1])

    # ── multi-recording file ──────────────────────────────────────────────────

    def _merged(self, *recordings):
        """Write each recording to its own H5, then merge all into one master file."""
        h5m = self.tmp_path / "merged.h5"
        with h5py.File(h5m, "w") as dst:
            for rec in recordings:
                src_path = self.tmp_path / f"{rec}.h5"
                self._write(src_path, rec)
                with h5py.File(src_path, "r") as src:
                    for key in src.keys():
                        src.copy(key, dst)
        return h5m

    def test_multi_recording_catalogue(self):
        h5m = self._merged("rec_a", "rec_b")
        self.assertEqual(sorted(lfpack.LFPackReader.recordings(h5m)), ["rec_a", "rec_b"])

    def test_multi_recording_read(self):
        h5m = self._merged("rec_a", "rec_b")
        for rec in ("rec_a", "rec_b"):
            sr = lfpack.LFPackReader(h5m, recording=rec)
            data, _ = sr.read_samples(0, self.NS)
            self.assertEqual(data.shape, (self.NS, self.NC))

    def test_auto_detect_raises_on_multi(self):
        h5m = self._merged("rec_a", "rec_b")
        with self.assertRaises(ValueError):
            lfpack.LFPackReader(h5m)

    def test_missing_recording_raises(self):
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        with self.assertRaises(KeyError):
            lfpack.LFPackReader(h5, recording="does_not_exist")


class TestLFPackReaderAPI(unittest.TestCase):
    """Edge-case paths in LFPackReader and compress_to_h5 not covered by the round-trip tests."""

    NC = 32
    NS = 512
    CHUNK = 256
    OVERLAP = 32

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        data = _synthetic_lfp(nc=self.NC, ns=self.NS)
        self.npy = self.tmp_path / "ck.npy"
        np.save(self.npy, data.T.astype(np.float32))
        self.h = {"x": np.zeros(self.NC, dtype=np.float32), "y": np.arange(self.NC, dtype=np.float32) * 25.0}
        self.h5 = self.tmp_path / "t.h5"
        lfpack.compress_to_h5(
            self.npy, self.h5, recording="r", h=self.h, chunk=self.CHUNK, overlap=self.OVERLAP, n_jobs=1
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_compress_to_h5_default_geometry(self):
        """h=None falls back to NP1 geometry (covers the h-is-None branch)."""
        h5 = self.tmp_path / "no_geom.h5"
        lfpack.compress_to_h5(self.npy, h5, recording="r", chunk=self.CHUNK, overlap=self.OVERLAP, n_jobs=1)
        sr = lfpack.LFPackReader(h5)
        self.assertEqual(sr.nc, self.NC)

    def test_close_is_open_is_mtscomp(self):
        sr = lfpack.LFPackReader(self.h5)
        self.assertTrue(sr.is_open)
        self.assertFalse(sr.is_mtscomp)
        self.assertGreater(sr.fs, 0)
        sr.close()
        self.assertFalse(sr.is_open)
        sr.close()  # idempotent

    def test_read_raises_when_closed(self):
        sr = lfpack.LFPackReader(self.h5)
        sr.close()
        with self.assertRaises(IOError):
            sr.read(slice(0, 10))

    def test_read_int_nsel(self):
        sr = lfpack.LFPackReader(self.h5)
        data, _ = sr.read(0)
        self.assertEqual(data.shape, (1, self.NC))

    def test_read_channel_selection(self):
        sr = lfpack.LFPackReader(self.h5)
        data, _ = sr.read(slice(0, 100), csel=slice(0, 4))
        self.assertEqual(data.shape, (100, 4))

    def test_read_no_sync(self):
        sr = lfpack.LFPackReader(self.h5)
        data = sr.read(slice(0, 10), sync=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (10, self.NC))

    def test_bin_channels_property_shapes(self):
        """Instantiating with bin_channels propagates to nc, shape, geometry, and slicing."""
        for n in (2, 4):
            sr = lfpack.LFPackReader(self.h5, bin_channels=n)
            self.assertEqual(sr.bin_channels, n)
            self.assertEqual(sr.nc, self.NC // n)
            self.assertEqual(sr.shape, (self.NS, self.NC // n))
            self.assertEqual(sr.geometry["x"].shape, (self.NC // n,))
            self.assertEqual(sr.geometry["y"].shape, (self.NC // n,))
            data = sr[0 : self.NS, :]
            self.assertEqual(data.shape, (self.NS, self.NC // n))

    def test_geometry_full_always_raw(self):
        """geometry_full returns full (nc,) arrays and binned_channel_index regardless of bin_channels."""
        sr = lfpack.LFPackReader(self.h5, bin_channels=4)
        gf = sr.geometry_full
        self.assertEqual(gf["x"].shape, (self.NC,))
        self.assertEqual(gf["y"].shape, (self.NC,))
        self.assertEqual(gf["binned_channel_index"].shape, (self.NC,))
        # channel i maps to bin i // 4
        np.testing.assert_array_equal(gf["binned_channel_index"], np.arange(self.NC) // 4)

    def test_read_samples_bin_channels_shape(self):
        """read_samples(bin_channels=N) returns (ns, nc//N)."""
        sr = lfpack.LFPackReader(self.h5)
        for n in (2, 4):
            data, _ = sr.read_samples(0, self.NS, bin_channels=n)
            self.assertEqual(data.shape, (self.NS, self.NC // n))
            self.assertEqual(data.dtype, np.float32)

    def test_read_samples_bin_channels_math(self):
        """Binned read equals column-sum of adjacent channels in the unbinned read."""
        sr = lfpack.LFPackReader(self.h5)
        full, _ = sr.read_samples(0, self.NS)  # (ns, nc)
        for n in (2, 4):
            binned, _ = sr.read_samples(0, self.NS, bin_channels=n)  # (ns, nc//n)
            nc_binned = self.NC // n
            expected = full[:, : nc_binned * n].reshape(self.NS, nc_binned, n).sum(axis=2)
            np.testing.assert_allclose(binned.astype(np.float64), expected.astype(np.float64), rtol=1e-4)

    def test_scales_missing_recording_raises(self):
        with self.assertRaises(KeyError):
            lfpack.LFPackReader.scales(self.h5, "no_such_recording")

    def test_legacy_format(self):
        """LFPackReader reads legacy flat layout (meta/chunks at root) and recordings() returns []."""
        leg = self.tmp_path / "legacy.h5"
        with h5py.File(self.h5, "r") as src, h5py.File(leg, "w") as dst:
            src.copy("r/00/meta", dst, name="meta")
            src.copy("r/00/chunks", dst, name="chunks")
        self.assertEqual(lfpack.LFPackReader.recordings(leg), [])
        sr = lfpack.LFPackReader(leg)
        data, _ = sr.read_samples(0, self.NS)
        self.assertEqual(data.shape, (self.NS, self.NC))


class TestCompressPipeline(unittest.TestCase):
    """Tests for compress_pipeline (Cadzow + SVD + WP end-to-end)."""

    NC = 32
    NS = 500

    def setUp(self):
        self.data = _synthetic_lfp(nc=self.NC, ns=self.NS)
        self.h = {"x": np.zeros(self.NC, dtype=np.float32), "y": np.arange(self.NC, dtype=np.float32) * 25.0}

    def test_compress_pipeline_shape(self):
        reconstructed, compressed = lfpack.compress_pipeline(self.data, h=self.h)
        self.assertEqual(reconstructed.shape, self.data.shape)
        self.assertEqual(reconstructed.dtype, np.float32)

    def test_compress_pipeline_cr(self):
        _, compressed = lfpack.compress_pipeline(self.data, h=self.h)
        self.assertGreater(compressed.cr_total, 1.0)

    def test_compress_pipeline_default_geometry(self):
        """h=None uses the NP1 geometry for nc channels."""
        reconstructed, _ = lfpack.compress_pipeline(self.data)
        self.assertEqual(reconstructed.shape, self.data.shape)


if __name__ == "__main__":
    unittest.main()
