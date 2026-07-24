# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `subset_h5` — copy a subset of recordings out of a multi-recording HDF5 archive
  (inverse of `merge_h5`), for carving a smaller release (e.g. BWM) out of a larger
  superset (e.g. ephys-atlas) without re-compression.
- Bad-channel labels (0=good, 1=dead, 2=noisy, 3=outside brain) persisted through
  compression and exposed via `LFPackReader.channels` / `channels_full`.
- **Saturation detection and muting**: ADC-clipped spans are detected on the raw LFP
  band, stored as a per-recording interval table, and muted (cosine-tapered zero) on
  the decimated output after Cadzow denoising. Exposed via `LFPackReader.saturation`
  (raw-rate sample indices, recording-aligned) and `.saturation_summary`.
  `.saturation_mask` is a property, indexed like the reader itself (`sr.saturation_mask[a:b]`);
  `.saturation_times()` converts samples to session-clock seconds (matching `times`).

### Fixed
- `compress` no longer decompresses low-SNR chunks to exact zero; new `floor_k=64`
  survival floor keeps the dominant mode's largest coefficients. Closes #2.

## [0.2.0] - 2026-06-30

### Added
- `LFPackReader.channels` and `LFPackReader.channels_full` — per-channel probe geometry
  and brain location annotations (`x/y/z` MNI coordinates, `atlas_id`, `acronym`).
  Brain location fields are optional; the properties work on any existing file.
  `channels` aggregates over binned-channel groups (mean for coordinates, mode for
  brain region); `channels_full` always returns the raw per-electrode data.
- `compress_to_h5` accepts an optional `channels` dict to embed brain location
  annotations at compression time.
