# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**lfpack** is a lossy LFP (local-field-potential) codec for Neuropixels recordings. It compresses raw `.cbin` LFP binaries into HDF5 files at >100× compression with RMSE < 25 µV, using an 8-stage pipeline followed by adaptive SVD + wavelet-packet thresholding.

## Commands

```bash
# Dev setup
uv sync --group dev
git config core.hooksPath .githooks   # installs ruff pre-commit hook

# Tests
uv run pytest                          # all tests
uv run pytest tests/test_lfpack.py::TestLfpack::test_compress_output_shapes  # single test

# Lint / format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Architecture

Nearly all code lives in `src/lfpack/_core.py` (~1000 lines). The public API is re-exported from `src/lfpack/__init__.py`.

### Compression pipeline (`compress_bin_to_h5`)

1. Bad-channel detection (`ibldsp.voltage.detect_bad_channels_cbin`)
2. Dephasing — sample-shift correction (NP1 only, via `ibldsp.fourier.fshift`)
3. Highpass filter — 2 Hz zero-phase 3rd-order Butterworth
4. Bad-channel interpolation — distance-weighted neighbors
5. CAR — median subtraction across channels
6. Decimation — 2500 → 250 Hz (Q=10)
7. Cadzow denoising — spatial rank reduction (`ibldsp.cadzow`)
8. **Adaptive SVD + wavelet-packet thresholding** — the actual codec

Steps 6–7 are checkpointed to a `.npy` file so the expensive decimation can be skipped on resume. Steps 7–8 are parallelised with `joblib`.

### Codec (`compress` / `decompress`)

`compress(data, epsilon=150, alpha=28)` → `LFPCompressed` dataclass:
- **SVD rank selection**: keep singular values > `epsilon × sigma_noise`
- **Wavelet-packet thresholding**: db4, level 5; per-component threshold `alpha × sigma_noise / sv[k]`
- Sparse Vh stored as `(vh_indices, vh_values)` in HDF5; `U_scaled` with shuffle+gzip

Guard bands: 64-sample Cadzow halos, 128-sample SVD/WP overlap to prevent edge transients.

### HDF5 layout (multi-recording, pyramidal)

```
<file>.h5
└─ <recording>/
   └─ <scale_2digit>/
      ├─ meta         # nc, ns_total, fs, epsilon, alpha, geometry, …
      └─ chunks/
         └─ <i>/      # U_scaled, vh_indices, vh_values + attrs
```

Legacy flat layout (meta at root) is still readable for backwards compatibility.

### `LFPackReader`

Drop-in replacement for `spikeglx.Reader`. Wraps an HDF5 file and decompresses chunks on demand. Supports multi-recording files via `recording=` kwarg and pyramidal scales via `scale=` kwarg.
