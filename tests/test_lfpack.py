import unittest

import numpy as np

import lfpack


def _synthetic_lfp(nc=32, ns=500):
    """Return a deterministic low-rank (nc, ns) float32 array that resembles LFP structure."""
    t = np.linspace(0, 4 * np.pi, ns)
    temporal = np.array([np.sin(t), np.cos(t), np.sin(2 * t)])  # (3, ns)
    # spatial weights: channel index modulated by three patterns, scaled to LFP µV range
    ch = np.arange(nc, dtype=np.float64)
    spatial = np.column_stack([
        np.sin(ch / nc * np.pi),
        np.cos(ch / nc * np.pi),
        np.sin(2 * ch / nc * np.pi),
    ]) * np.array([100.0, 30.0, 10.0])  # (nc, 3)
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


if __name__ == '__main__':
    unittest.main()
