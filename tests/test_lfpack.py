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

    def test_survival_floor_prevents_all_zero(self):
        """A chunk that over-thresholds to all-zero (floor_k=0) is rescued by the floor."""
        # A huge alpha thresholds every WP coefficient, reproducing the low-SNR pathology.
        c0 = lfpack.compress(self.data, alpha=1e6, floor_k=0)
        cf = lfpack.compress(self.data, alpha=1e6, floor_k=64)
        self.assertTrue(np.all(lfpack.decompress(c0) == 0), "floor_k=0 should decompress to all-zero")
        rec = lfpack.decompress(cf)
        self.assertFalse(np.all(rec == 0), "survival floor must keep the reconstruction non-zero")
        # dominant mode keeps floor_k coefficients (one row floored, not every row)
        self.assertEqual(int(np.count_nonzero(cf.Vh_hat)), 64)

    def test_survival_floor_zero_input_stays_zero(self):
        """All-zero (saturation-muted) input decompresses to exact zero despite the floor."""
        z = np.zeros_like(self.data)
        rec = lfpack.decompress(lfpack.compress(z, floor_k=64))
        self.assertTrue(np.all(rec == 0))

    def test_survival_floor_no_op_when_row_keeps_enough(self):
        """When the dominant row already keeps >= floor_k coeffs the floor never fires,
        so the reconstruction is bit-for-bit identical (the real-LFP high-SNR case)."""
        c0 = lfpack.compress(self.data, alpha=1.0, floor_k=0)
        self.assertGreaterEqual(int(np.count_nonzero(c0.Vh_hat[0])), 64)  # config sanity
        r0 = lfpack.decompress(c0)
        r64 = lfpack.decompress(lfpack.compress(self.data, alpha=1.0, floor_k=64))
        self.assertTrue(np.array_equal(r0, r64))

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

    def test_compress_to_h5_libver(self):
        """compress_to_h5 must not use libver='latest' (moving target — breaks cross-version reads)."""
        h5 = self.tmp_path / "single.h5"
        self._write(h5, "rec_a")
        with open(h5, "rb") as fh:
            fh.seek(8)
            superblock_version = fh.read(1)[0]
        self.assertLess(superblock_version, 3, "compress_to_h5 must not use libver='latest' (moving target)")

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

    def _annotate_h5(self):
        """Write brain-location attrs to self.h5 and return the injected arrays."""
        ml = np.linspace(-3e-3, 3e-3, self.NC, dtype=np.float32)
        ap = np.zeros(self.NC, dtype=np.float32)
        dv = np.linspace(-5e-3, 0, self.NC, dtype=np.float32)
        # atlas_id: four groups of NC//4 channels each get a distinct id
        n = 4
        atlas_id = np.repeat(np.arange(self.NC // n, dtype=np.int32), n)
        acronym = [f"reg{i // n}" for i in range(self.NC)]
        # labels: bad-channel quality flag, constant within each bin group so the
        # within-group mode is well defined (cycles through 0=good..3=outside).
        labels = np.repeat(np.arange(self.NC // n, dtype=np.int8) % 4, n)
        with h5py.File(self.h5, "a") as f:
            meta = f["r/00/meta"]
            meta.attrs["ml"] = ml
            meta.attrs["ap"] = ap
            meta.attrs["dv"] = dv
            meta.attrs["atlas_id"] = atlas_id
            meta.attrs["acronym"] = acronym
            meta.attrs["labels"] = labels
        return ml, ap, dv, atlas_id, acronym, labels

    def test_channels_no_annotation(self):
        """channels / channels_full work without brain-location attrs (optional fields)."""
        for prop in ("channels", "channels_full"):
            ch = getattr(lfpack.LFPackReader(self.h5), prop)
            self.assertEqual(ch["lateral_um"].shape, (self.NC,))
            self.assertEqual(ch["axial_um"].shape, (self.NC,))
            self.assertNotIn("x", ch)
            self.assertNotIn("atlas_id", ch)

    def test_channels_full_roundtrip(self):
        """channels_full returns raw (nc,) arrays regardless of bin_channels."""
        sr = lfpack.LFPackReader(self.h5)
        sr.close()
        ml, ap, dv, atlas_id, acronym, labels = self._annotate_h5()
        for bin_ch in (1, 4):
            ch = lfpack.LFPackReader(self.h5, bin_channels=bin_ch).channels_full
            np.testing.assert_array_almost_equal(ch["x"], ml)
            np.testing.assert_array_almost_equal(ch["z"], dv)
            self.assertEqual(ch["atlas_id"].shape, (self.NC,))
            self.assertEqual(ch["acronym"], acronym)
            np.testing.assert_array_equal(ch["labels"], labels)

    def test_channels_binned_aggregation(self):
        """channels aggregates float fields by mean and categorical fields by mode."""
        sr = lfpack.LFPackReader(self.h5)
        sr.close()
        ml, ap, dv, atlas_id, acronym, labels = self._annotate_h5()
        n = 4
        sr_b = lfpack.LFPackReader(self.h5, bin_channels=n)
        ch = sr_b.channels
        nc_b = self.NC // n
        # float fields averaged
        self.assertEqual(ch["lateral_um"].shape, (nc_b,))
        np.testing.assert_array_almost_equal(ch["x"], ml.reshape(nc_b, n).mean(axis=1))
        # categorical fields: mode per bin (each group has a single atlas_id)
        self.assertEqual(ch["atlas_id"].shape, (nc_b,))
        np.testing.assert_array_equal(ch["atlas_id"], np.arange(nc_b, dtype=np.int32))
        self.assertEqual(ch["acronym"], [f"reg{i}" for i in range(nc_b)])
        # labels: mode per bin (each group is a single label value)
        self.assertEqual(ch["labels"].shape, (nc_b,))
        np.testing.assert_array_equal(ch["labels"], np.arange(nc_b, dtype=np.int8) % 4)

    def test_fs_without_sync_returns_nominal(self):
        """fs falls back to the nominal rate stored in meta when no sync data."""
        sr = lfpack.LFPackReader(self.h5)
        self.assertFalse(np.isnan(sr.fs))
        self.assertGreater(sr.fs, 0)
        self.assertEqual(sr.fs, sr._fs)

    def test_t0_without_sync_returns_nan(self):
        """t0 is NaN when the file has no sync attributes."""
        sr = lfpack.LFPackReader(self.h5)
        self.assertTrue(np.isnan(sr.t0))

    def test_fs_and_t0_with_sync(self):
        """fs returns fs_sync and t0 returns t0_sync when sync attrs are present."""
        t0_ref = 123.456
        fs_ref = 249.997
        h5_sync = self.tmp_path / "sync.h5"
        lfpack.compress_to_h5(
            self.npy,
            h5_sync,
            recording="r",
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            n_jobs=1,
            t0_sync=t0_ref,
            fs_sync=fs_ref,
        )
        sr = lfpack.LFPackReader(h5_sync)
        self.assertAlmostEqual(sr.fs, fs_ref, places=6)
        self.assertAlmostEqual(sr.t0, t0_ref, places=6)

    def test_times_without_sync(self):
        """times starts at 0 and is spaced by 1/fs when no sync data."""
        sr = lfpack.LFPackReader(self.h5)
        t = sr.times
        self.assertEqual(t.shape, (self.NS,))
        self.assertAlmostEqual(float(t[0]), 0.0, places=9)
        self.assertAlmostEqual(float(t[1] - t[0]), 1.0 / sr.fs, places=9)

    def test_times_with_sync(self):
        """times starts at t0_sync and is spaced by 1/fs_sync."""
        t0_ref, fs_ref = 100.0, 249.5
        h5_sync = self.tmp_path / "sync2.h5"
        lfpack.compress_to_h5(
            self.npy,
            h5_sync,
            recording="r",
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            n_jobs=1,
            t0_sync=t0_ref,
            fs_sync=fs_ref,
        )
        sr = lfpack.LFPackReader(h5_sync)
        t = sr.times
        self.assertEqual(t.shape, (self.NS,))
        self.assertAlmostEqual(float(t[0]), t0_ref, places=9)
        self.assertAlmostEqual(float(t[1] - t[0]), 1.0 / fs_ref, places=9)

    def test_scales_skips_sync_group(self):
        """scales() must not choke on non-numeric groups like a future sync group."""
        with h5py.File(self.h5, "a") as f:
            f.require_group("r/sync")  # inject a non-numeric sibling
        scales = lfpack.LFPackReader.scales(self.h5, "r")
        self.assertEqual(scales, [0])


class TestMergeH5(unittest.TestCase):
    """Tests for merge_h5: multi-recording aggregation without re-compression."""

    NC = 32
    NS = 512
    CHUNK = 256
    OVERLAP = 32

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        data = _synthetic_lfp(nc=self.NC, ns=self.NS)
        npy = self.tmp_path / "ck.npy"
        np.save(npy, data.T.astype(np.float32))
        self.h = {"x": np.zeros(self.NC, dtype=np.float32), "y": np.arange(self.NC, dtype=np.float32) * 25.0}
        self.npy = npy

    def tearDown(self):
        self.tmp.cleanup()

    def _make_h5(self, recording):
        """Compress the shared npy into a fresh single-recording H5 and return its path."""
        h5 = self.tmp_path / f"{recording}.h5"
        lfpack.compress_to_h5(
            self.npy, h5, recording=recording, h=self.h, chunk=self.CHUNK, overlap=self.OVERLAP, n_jobs=1
        )
        return h5

    def test_recordings_in_merged_file(self):
        """merge_h5 produces a file with exactly the expected recording names."""
        h5a = self._make_h5("rec_a")
        h5b = self._make_h5("rec_b")
        merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a, h5b], merged)
        self.assertEqual(sorted(lfpack.LFPackReader.recordings(merged)), ["rec_a", "rec_b"])

    def test_hierarchy_preserved(self):
        """The full /<rec>/00/meta and /<rec>/00/chunks/<i>/ structure survives the copy."""
        h5a = self._make_h5("rec_a")
        merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a], merged)
        with h5py.File(merged, "r") as f:
            # top-level group and scale sub-group
            self.assertIn("rec_a", f)
            self.assertIn("00", f["rec_a"])
            # meta attrs
            meta = f["rec_a/00/meta"]
            for attr in ("nc", "ns_total", "fs", "compress_chunk", "compress_overlap"):
                self.assertIn(attr, meta.attrs)
            self.assertEqual(int(meta.attrs["nc"]), self.NC)
            self.assertEqual(int(meta.attrs["ns_total"]), self.NS)
            # at least one chunk with the expected datasets
            chunks = f["rec_a/00/chunks"]
            self.assertGreater(len(chunks), 0)
            chunk0 = chunks["0"]
            for ds in ("U_scaled", "vh_indices", "vh_values"):
                self.assertIn(ds, chunk0)
            for attr in ("ns_original", "vh_shape", "cr_total", "rmse"):
                self.assertIn(attr, chunk0.attrs)

    def test_merge_h5_libver(self):
        """Merged file must not use libver='latest' (moving target — breaks cross-version reads)."""
        h5a = self._make_h5("rec_a")
        merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a], merged)
        # libver=('earliest', 'v110') produces superblock v0; libver='latest' produces v3 which is a moving target
        with open(merged, "rb") as fh:
            fh.seek(8)
            superblock_version = fh.read(1)[0]
        self.assertLess(superblock_version, 3, "merge_h5 must not use libver='latest' (moving target)")

    def test_merged_file_is_readable(self):
        """LFPackReader can decompress data from a merged file."""
        h5a = self._make_h5("rec_a")
        h5b = self._make_h5("rec_b")
        merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a, h5b], merged)
        for rec in ("rec_a", "rec_b"):
            sr = lfpack.LFPackReader(merged, recording=rec)
            data, _ = sr.read_samples(0, self.NS)
            self.assertEqual(data.shape, (self.NS, self.NC))

    def test_duplicate_recording_name_raises(self):
        """Two source files that map to the same recording name raise ValueError."""
        h5a = self._make_h5("rec_a")
        h5b = self._make_h5("rec_b")
        merged = self.tmp_path / "merged.h5"
        with self.assertRaises(ValueError):
            lfpack.merge_h5([h5a, h5b], merged, recording_map={h5b: "rec_a"})

    def test_duplicate_within_src_files_raises(self):
        """Two source files with the same internal key raise ValueError (no recording_map needed)."""
        h5a = self._make_h5("same_key")
        h5b = self.tmp_path / "same_key_2.h5"
        lfpack.compress_to_h5(
            self.npy, h5b, recording="same_key", h=self.h, chunk=self.CHUNK, overlap=self.OVERLAP, n_jobs=1
        )
        merged = self.tmp_path / "merged.h5"
        with self.assertRaises(ValueError):
            lfpack.merge_h5([h5a, h5b], merged)

    def test_recording_map_renames_key(self):
        """recording_map lets the caller override the recording name in the merged file."""
        h5a = self._make_h5("original_name")
        merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a], merged, recording_map={h5a: "renamed"})
        self.assertEqual(lfpack.LFPackReader.recordings(merged), ["renamed"])
        # original name must not appear
        with h5py.File(merged, "r") as f:
            self.assertNotIn("original_name", f)

    def test_multi_file_raises_on_multi_recording_source(self):
        """A source file with more than one top-level group raises ValueError."""
        h5a = self._make_h5("rec_a")
        h5b = self._make_h5("rec_b")
        # build a file that already has two recordings (like a previously merged file)
        multi = self.tmp_path / "multi.h5"
        lfpack.merge_h5([h5a, h5b], multi)
        merged = self.tmp_path / "merged2.h5"
        with self.assertRaises(ValueError):
            lfpack.merge_h5([multi], merged)

    def test_returns_resolved_path(self):
        """merge_h5 returns the resolved Path to the output file."""
        h5a = self._make_h5("rec_a")
        merged = self.tmp_path / "merged.h5"
        result = lfpack.merge_h5([h5a], merged)
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())


class TestSubsetH5(unittest.TestCase):
    """Tests for subset_h5: pulling a subset of recordings out of a merged archive."""

    NC = 32
    NS = 512
    CHUNK = 256
    OVERLAP = 32

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        data = _synthetic_lfp(nc=self.NC, ns=self.NS)
        npy = self.tmp_path / "ck.npy"
        np.save(npy, data.T.astype(np.float32))
        self.h = {"x": np.zeros(self.NC, dtype=np.float32), "y": np.arange(self.NC, dtype=np.float32) * 25.0}
        self.npy = npy

        h5a = self.tmp_path / "rec_a.h5"
        h5b = self.tmp_path / "rec_b.h5"
        h5c = self.tmp_path / "rec_c.h5"
        for h5, rec in ((h5a, "rec_a"), (h5b, "rec_b"), (h5c, "rec_c")):
            lfpack.compress_to_h5(
                self.npy, h5, recording=rec, h=self.h, chunk=self.CHUNK, overlap=self.OVERLAP, n_jobs=1
            )
        self.merged = self.tmp_path / "merged.h5"
        lfpack.merge_h5([h5a, h5b, h5c], self.merged)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selected_recordings_present(self):
        """subset_h5 copies exactly the requested recordings, nothing else."""
        dst = self.tmp_path / "subset.h5"
        lfpack.subset_h5(self.merged, dst, ["rec_a", "rec_c"])
        self.assertEqual(sorted(lfpack.LFPackReader.recordings(dst)), ["rec_a", "rec_c"])

    def test_subset_is_readable(self):
        """LFPackReader can decompress data from a subset file."""
        dst = self.tmp_path / "subset.h5"
        lfpack.subset_h5(self.merged, dst, ["rec_b"])
        sr = lfpack.LFPackReader(dst, recording="rec_b")
        data, _ = sr.read_samples(0, self.NS)
        self.assertEqual(data.shape, (self.NS, self.NC))

    def test_missing_recording_raises_by_default(self):
        """A requested recording absent from the source raises ValueError."""
        dst = self.tmp_path / "subset.h5"
        with self.assertRaises(ValueError):
            lfpack.subset_h5(self.merged, dst, ["rec_a", "not_there"])

    def test_missing_recording_warn_skips(self):
        """missing='warn' skips absent recordings instead of raising."""
        dst = self.tmp_path / "subset.h5"
        lfpack.subset_h5(self.merged, dst, ["rec_a", "not_there"], missing="warn")
        self.assertEqual(lfpack.LFPackReader.recordings(dst), ["rec_a"])

    def test_invalid_missing_value_raises(self):
        """An unrecognised missing= value raises ValueError."""
        dst = self.tmp_path / "subset.h5"
        with self.assertRaises(ValueError):
            lfpack.subset_h5(self.merged, dst, ["rec_a"], missing="ignore")

    def test_returns_resolved_path(self):
        """subset_h5 returns the resolved Path to the output file."""
        dst = self.tmp_path / "subset.h5"
        result = lfpack.subset_h5(self.merged, dst, ["rec_a"])
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())


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


class TestSaturationTable(unittest.TestCase):
    """Saturation intervals storage in compress_to_h5 and the LFPackReader access API."""

    NC = 32
    NS = 1000  # decimated (reader-rate) samples
    CHUNK = 256
    OVERLAP = 32
    FS_RAW = 2500.0  # raw LFP rate the intervals are stored at
    FS_DEC = 250.0  # reader (decimated) rate → ratio 1/10

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        np.save(self.tmp_path / "chk.npy", _synthetic_lfp(self.NC, self.NS).T.astype(np.float32))
        self.npy = self.tmp_path / "chk.npy"
        self.h = {"x": np.zeros(self.NC, dtype=np.float32), "y": np.arange(self.NC, dtype=np.float32) * 25.0}
        # two saturated spans at the raw rate; ns_total = NS * 10 = 10000 raw samples
        self.intervals = np.array([[1000, 1500], [8000, 8200]], dtype=np.int64)
        n_sat = int((self.intervals[:, 1] - self.intervals[:, 0]).sum())  # 700
        self.attrs = {
            "fs": self.FS_RAW,
            "ns_total": self.NS * 10,
            "n_saturated_samples": n_sat,
            "saturated_fraction": n_sat / (self.NS * 10),
            "v_per_sec": 1e-8,
            "proportion": 0.2,
            "mute_window_samples": 7,
            "muted": True,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, h5_path, intervals=None, attrs=None):
        lfpack.compress_to_h5(
            self.npy,
            h5_path,
            recording="rec_a",
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            fs=self.FS_DEC,
            n_jobs=1,
            saturation_intervals=self.intervals if intervals is None else intervals,
            saturation_attrs=self.attrs if attrs is None else attrs,
        )
        return h5_path

    def test_dataset_written_at_recording_level(self):
        h5 = self._write(self.tmp_path / "s.h5")
        with h5py.File(h5, "r") as f:
            self.assertIn("rec_a/saturation", f)
            self.assertNotIn("rec_a/00/saturation", f)  # scale-independent
            np.testing.assert_array_equal(f["rec_a/saturation"][()], self.intervals)

    def test_reader_saturation_table(self):
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "s.h5"))
        df = sr.saturation
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["start_sample", "stop_sample"])
        np.testing.assert_array_equal(df["start_sample"].to_numpy(), self.intervals[:, 0])
        np.testing.assert_array_equal(df["stop_sample"].to_numpy(), self.intervals[:, 1])

    def test_reader_summary(self):
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "s.h5"))
        s = sr.saturation_summary
        self.assertEqual(s["n_intervals"], 2)
        self.assertEqual(s["n_saturated_samples"], 700)
        self.assertAlmostEqual(s["saturated_fraction"], 700 / 10000)
        self.assertAlmostEqual(s["total_saturated_sec"], 700 / self.FS_RAW)
        self.assertTrue(s["muted"])

    def test_mask_converts_to_reader_rate(self):
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "s.h5"))
        mask = sr.saturation_mask
        self.assertEqual(mask.shape, (self.NS,))
        # raw [1000,1500) → dec [100,150); raw [8000,8200) → dec [800,820)
        self.assertTrue(mask[100:150].all())
        self.assertTrue(mask[800:820].all())
        self.assertFalse(mask[:100].any())
        self.assertFalse(mask[150:800].any())
        self.assertEqual(int(mask.sum()), 70)

    def test_mask_is_indexable_like_reader(self):
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "s.h5"))
        # sliced exactly like sr[...] itself, not called with window arguments
        self.assertTrue(sr.saturation_mask[100:150].all())
        self.assertFalse(sr.saturation_mask[150:800].any())

    def test_mask_windowed_and_outward_rounding(self):
        # a span that does not land on a decimated-sample boundary must round outward
        odd = np.array([[1005, 1006]], dtype=np.int64)  # raw → dec [100.5, 100.6) → [100,101)
        sr2 = lfpack.LFPackReader(self._write(self.tmp_path / "o.h5", intervals=odd))
        m = sr2.saturation_mask[90:120]
        self.assertEqual(m.shape, (30,))
        self.assertTrue(m[10])  # dec sample 100 → offset 10 in the window
        self.assertEqual(int(m.sum()), 1)

    def test_empty_intervals(self):
        empty = np.zeros((0, 2), dtype=np.int64)
        attrs = {**self.attrs, "n_saturated_samples": 0, "saturated_fraction": 0.0}
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "e.h5", intervals=empty, attrs=attrs))
        self.assertTrue(sr.saturation.empty)
        self.assertEqual(sr.saturation_summary["n_intervals"], 0)
        self.assertFalse(sr.saturation_mask.any())

    def test_backward_compatible_when_absent(self):
        """Files written without saturation detection expose empty table/default summary."""
        h5 = self.tmp_path / "none.h5"
        lfpack.compress_to_h5(
            self.npy,
            h5,
            recording="rec_a",
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            fs=self.FS_DEC,
            n_jobs=1,
        )
        sr = lfpack.LFPackReader(h5)
        self.assertTrue(sr.saturation.empty)
        self.assertEqual(sr.saturation_summary["saturated_fraction"], 0.0)
        self.assertFalse(sr.saturation_mask.any())

    def test_saturation_times_session_clock(self):
        """saturation_times converts samples to session-clock seconds via t0/fs, not raw fs."""
        t0, fs_sync = 1000.0, 250.5
        h5 = self.tmp_path / "s.h5"
        lfpack.compress_to_h5(
            self.npy,
            h5,
            recording="rec_a",
            h=self.h,
            chunk=self.CHUNK,
            overlap=self.OVERLAP,
            fs=self.FS_DEC,
            t0_sync=t0,
            fs_sync=fs_sync,
            n_jobs=1,
            saturation_intervals=self.intervals,
            saturation_attrs=self.attrs,
        )
        sr = lfpack.LFPackReader(h5)
        df = sr.saturation_times()
        self.assertEqual(list(df.columns), ["start_sample", "stop_sample", "start_time", "stop_time"])
        ratio = self.FS_DEC / self.FS_RAW
        expected_start0 = t0 + self.intervals[0, 0] * ratio / fs_sync
        self.assertAlmostEqual(df["start_time"].iloc[0], expected_start0)
        # falls back to t0=0 when no sync data is present
        sr_nosync = lfpack.LFPackReader(self._write(self.tmp_path / "s2.h5"))
        df2 = sr_nosync.saturation_times()
        self.assertAlmostEqual(df2["start_time"].iloc[0], self.intervals[0, 0] * ratio / self.FS_DEC)

    def test_saturation_times_empty(self):
        empty = np.zeros((0, 2), dtype=np.int64)
        attrs = {**self.attrs, "n_saturated_samples": 0, "saturated_fraction": 0.0}
        sr = lfpack.LFPackReader(self._write(self.tmp_path / "e.h5", intervals=empty, attrs=attrs))
        df = sr.saturation_times()
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["start_sample", "stop_sample", "start_time", "stop_time"])


if __name__ == "__main__":
    unittest.main()
