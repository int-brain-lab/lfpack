# lfpack — LFP codec for Neuropixels recordings

<p align="center">
  <img src="docs/figures/logo.png" alt="lfpack logo" width="300"/>
</p>

Lossy codec for local-field-potential (LFP) recordings from Neuropixels probes.
Four-stage pipeline: decimation → Cadzow denoising → adaptive SVD → wavelet-packet
thresholding. Achieves **>100× compression** with median RMSE < 25 µV.

---

## Installation

```bash
pip install lfpack
```

or with uv:

```bash
uv add lfpack
```

Development install:

```bash
git clone https://github.com/int-brain-lab/lfpack
cd lfpack
uv sync --extra dev
```

---

## Getting started

```python
from lfpack import compress_bin_to_h5, LFPackReader

# Decimate, denoise, and compress a raw LFP binary in one call
compress_bin_to_h5('path/to/lf.cbin', 'lf.h5')

# Decode on demand — same interface as spikeglx.Reader
sr = LFPackReader('lf.h5')
traces = sr[0:1000]      # (1000, n_channels) float32, volts
geometry = sr.geometry   # {'x': ..., 'y': ...} channel positions
```

---

# LFP compression — methods and notes

## Pipeline overview

Four sequential stages applied to raw NP1/NP2 LFP data (384 ch, 2500 Hz):

```
raw LFP (2500 Hz)
  └─ 1. Decimation        → 250 Hz
  └─ 2. Cadzow denoising  → spatially denoised LFP
  └─ 3. SVD compression   → rank-r approximation
  └─ 4. WP thresholding   → sparse coefficient storage
```

---

## Stage 1 — Decimation (2500 → 250 Hz)

IIR anti-aliasing (`scipy.signal.decimate`, Q=10) applied in overlapping 30 s chunks
(0.2 s IIR guard halo each side) to avoid edge transients.

---

## Stage 2 — Cadzow denoising

Spatial denoising via the Cadzow algorithm (`ibldsp.cadzow.cadzow_denoiser`) run in
640-sample chunks with 64-sample halos (processed window = 768 = 3 × 256, FFT-optimal).

Parameters used: `rank=5, niter=1, fmax=None, nswx=64, gap_threshold=2.0, ppca_k=2.0`

---

## Stage 3 — Adaptive SVD

Each 2048-sample chunk is extended by 128-sample guard bands on each side before SVD.
Rank *r* is selected adaptively:

```
r = #{k : sv[k] > epsilon × sigma_noise}
```

where `sigma_noise` is the median of the upper half of the singular-value distribution
(restricted to `sv > 1e-4 × sv[0]` to exclude dead channels).

The stored quantity is `U_scaled = U[:, :r] * sv[:r]`  (shape `nc × r`).

---

## Stage 4 — Wavelet-packet thresholding

Each of the *r* temporal rows `Vh[k, :]` is decomposed with a db4 wavelet-packet tree
(level 5).  Per-component hard thresholding:

```
tau_k = alpha × sigma_noise / sv[k]
```

Larger singular values (dominant spatial modes) receive a lower threshold, preserving
more temporal detail.

### Theoretical compression-ratio formula

```
cr_total = (nc × ns) / (r × nc + n_kept)
cr_wp    = (r × ns)  / n_kept              # WP contribution to Vh rows only
cr_svd   = (nc × ns) / (r × (nc + ns))    # SVD-only CR (no WP)
```

---

## HDF5 storage optimisations

### Sparse Vh_hat

Rather than storing the full `(r, n_wp_slots)` float32 array and relying on gzip to
compress zero runs, Vh_hat is stored as two contiguous 1-D datasets:

| Dataset | dtype | length | description |
|---|---|---|---|
| `vh_indices` | int32 | n_kept | flat indices of non-zero coefficients |
| `vh_values` | float32 | n_kept | corresponding coefficient values |

`vh_shape` is stored as a group attribute so the dense array can be reconstructed on
read.

### Shuffle filter on U_scaled

The HDF5 `shuffle` filter is applied to `U_scaled` before gzip.  It reorders bytes by
significance (all MSBs together, then mantissa bytes), improving gzip compression on
dense float32 data.

---

## Results — flagship PID `eebcaf65` (76.5 min, 384 ch, NP1)

### Parameter sets

Original data: 2500 Hz, Recording length: 4588.45 s (76.47 min), 11,471,120 samples.
Compression ratios are relative to the original int16 binary (nc × ns × 2 bytes = 8.2 GB).

| Parameter set | ε (SVD) | α (WP) | File size | CR vs int16 binary | RMSE median | RMSE p95 |
|---|---|---|---|--------------------|---|---|
| **default** | 150 | 28 | 22 MB | **380×**           | 19 µV | 23 µV |
| **aggressive** | 450 | 96 | 11 MB | **760×**           | 34 µV | 40 µV |

### Density display — original, Cadzow, compressed, residuals

![Density display](docs/figures/density.png)

*2×5 panel: rows = default / aggressive; columns = original resampled → Cadzow →
Cadzow−original → compressed → compressed−Cadzow.  Single shared colormap ±190 µV.*

### Average PSD (600 s, nperseg=64)

![PSD comparison](docs/figures/psd.png)

*600 s of data, nperseg=256 (~1 Hz resolution, ~1200 Welch segments).
Solid lines: signal spectra.  Dashed: residuals after each processing step.
Default compression (orange) follows Cadzow closely to 125 Hz.
Aggressive (red) rolls off above ~80 Hz, sacrificing high-frequency content
for 2× smaller files.*

### RMSE distribution and CR vs RMSE scatter

![RMSE / CR scatter](docs/figures/rmse_cr.png)

### Storage breakdown (default parameters)

| Dataset | Stored | Gzip ratio |
|---|---|---|
| U_scaled (dense, shuffle+gzip) | ~11 MB | ~1.2× |
| Vh_hat sparse (indices + values, gzip) | ~5.4 MB | — |
| Info-theoretic minimum (U + n_kept floats) | ~17.8 MB | — |

U_scaled dominates (~50% of file) because dense float32 compresses poorly.