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

# Docs — regenerate API reference from docstrings, then preview
uv run quartodoc build --config docs/_quarto.yml
quarto preview docs/
```

## Architecture

Nearly all code lives in `src/lfpack/_core.py` (~1000 lines). The public API is re-exported from `src/lfpack/__init__.py`.

### Compression pipeline (`compress_bin_to_h5`)

0. Saturation detection — flag ADC-clipped samples on the raw LFP band *before* any step below obscures them (`ibldsp.voltage.saturation_cbin`, parallel over `n_jobs`). Detection is **amplitude-only** for LFP (`v_per_sec=None`): the derivative criterion is tuned for the 30 kHz AP band and mislabels normal LFP dynamics as saturation, so it is disabled here. Stored as an insertion-level interval table (see HDF5 layout) and used to mute the clipped stretches; toggle with `detect_saturation` / `saturation_kwargs`.
1. Bad-channel detection (`ibldsp.voltage.detect_bad_channels_cbin`) — labels saved to the scale-`00` `meta` as a `labels` int8 attr; read via `LFPackReader.channels`/`channels_full` under the `labels` key.
2. Dephasing — sample-shift correction (NP1 only, via `ibldsp.fourier.fshift`)
3. Highpass filter — 0.5 Hz zero-phase 3rd-order Butterworth (keeps delta/infra-slow; ibldsp warmup padding scales with the corner)
4. Bad-channel interpolation — distance-weighted neighbors
5. CAR — median subtraction across channels
6. Decimation — 2500 → 250 Hz (Q=10)
7. Cadzow denoising — spatial rank reduction (`ibldsp.cadzow`)
8. **Adaptive SVD + wavelet-packet thresholding** — the actual codec

Steps 6–7 are checkpointed to a `.npy` file so the expensive decimation can be skipped on resume. Steps 7–8 are parallelised with `joblib`. Muting (step 0): a raw-rate boolean saturation mask is passed to `resample_denoise_lfp_cbin` as `saturation_file`; each decimation worker rebuilds the cosine taper (`mute_window_samples`) and multiplies its raw slice **before** the highpass/anti-alias FIR, so rail values never ring into the passband. Muting is skipped when resuming from an existing checkpoint (data already decimated); the interval table is still written.

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
   ├─ saturation   # (n_intervals, 2) int64 [start_sample, stop_sample] at raw LFP fs;
   │               # written once per recording (scale-independent). attrs: fs, ns_total,
   │               # n_saturated_samples, saturated_fraction, detection params, muted
   └─ <scale_2digit>/
      ├─ meta         # nc, ns_total, fs, fs_sync, t0_sync, epsilon, alpha, geometry, …
      └─ chunks/
         └─ <i>/      # U_scaled, vh_indices, vh_values + attrs
```

The `saturation` node sits at recording level (not under a scale) because it describes the raw recording, not a codec pyramid level; `merge_h5` copies it automatically. Legacy flat layout (meta at root) is still readable for backwards compatibility.

### `LFPackReader`

Drop-in replacement for `spikeglx.Reader`. Wraps an HDF5 file and decompresses chunks on demand. Supports multi-recording files via `recording=` kwarg and pyramidal scales via `scale=` kwarg. Saturation access: `.saturation` (interval DataFrame with raw samples + seconds), `.saturation_summary` (fraction/count/muted from attrs), `saturation_mask(first, last)` (boolean at the reader's decimated rate, interval edges rounded outward). All three degrade gracefully to empty/default on files written without saturation detection.

## Ecosystem interconnections

lfpack sits in a tight three-way dependency with two sibling IBL packages:

- **ibl-neuropixel** (`/Users/olivier/PycharmProjects/ephys-atlas/ibl-neuropixel`) — provides the low-level Neuropixels binary I/O (`spikeglx.Reader`), destriping, dephasing (`ibldsp.fourier.fshift`), decimation (`ibldsp.voltage.resample_denoise_lfp_cbin`), bad-channel detection (`ibldsp.voltage.detect_bad_channels_cbin`), saturation detection (`ibldsp.voltage.saturation_cbin` / `saturation` / `saturation_samples_to_intervals`), and Cadzow denoising (`ibldsp.cadzow`). lfpack wraps these directly; changes to ibldsp can break the pipeline. `resample_denoise_lfp_cbin` gained a `saturation_file` / `mute_window_samples` parameter (added for this codec) to mute saturated raw samples pre-filter.
- **viewephys** (`/Users/olivier/PycharmProjects/ephys-atlas/viewephys`) — Qt-based interactive viewer for raw Neuropixels traces. Since [PR #49](https://github.com/int-brain-lab/viewephys/pull/49) it has a native lfpack backend (`LFPackDataModel` / `LFPackBinViewer`, optional `viewephys[lfpack]` extra): `viewephys -f file.h5` opens a `.h5` file directly, with automatic brain-region colouring from embedded `atlas_id`/`acronym` annotations, a searchable multi-recording selector, and a CSD step. No manual transpose or `BrainRegions` wiring needed.
