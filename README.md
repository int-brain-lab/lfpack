# lfpack — LFP codec for Neuropixels recordings

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
from lfpack import compress_to_h5, LFPCompressedReader

# Encode a Cadzow-denoised LFP array (.npy, shape (ns, nc)) to HDF5
compress_to_h5('lf_cadzow.npy', 'lf.h5')

# Decode on demand — same interface as spikeglx.Reader
reader = LFPCompressedReader('lf.h5')
traces = reader[0:1000]      # (1000, n_channels) float32, volts
geometry = reader.geometry   # {'x': ..., 'y': ...} channel positions
```

Low-level chunk API:

```python
from lfpack import compress, decompress
import numpy as np

snippet = np.random.randn(384, 2048).astype(np.float32)  # (nc, ns)
compressed = compress(snippet)
reconstructed = decompress(compressed)  # (nc, ns) float32
print(f'CR={compressed.cr_total:.0f}  RMSE={compressed.cr_svd:.1f}')
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

## Results — flagship PID `eebcaf65` (76 min, 384 ch)

| File | Size |
|---|---|
| `lf_resampled_car.npy` (raw decimated, float16) | 881 MB |
| `lf_resampled_car_cadzow.npy` (float32) | 1762 MB |
| `lf_compressed.h5` (this work) | **21.8 MB** |

True compression ratio: **114×** (original float32 / stored floats).
On-disk ratio vs raw decimated float32: **81×** (1762 MB → 21.8 MB).
RMSE per chunk: 15–24 µV (on Cadzow-denoised signal in volts).

### Storage breakdown

| Dataset | Stored | Gzip ratio |
|---|---|---|
| U_scaled (dense, shuffle+gzip) | ~11 MB | ~1.2× |
| Vh_hat sparse (indices + values, gzip) | ~5.4 MB | — |
| Info-theoretic minimum (U + n_kept floats) | ~17.8 MB | — |

U_scaled dominates (50% of file) because it is dense float data that compresses poorly.