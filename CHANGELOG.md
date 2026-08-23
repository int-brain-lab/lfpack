# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Non-linear sync support: `sync_samples`/`sync_times` — the raw ALF sync knot pairs
  (sample index <-> time), stored verbatim per scale — alongside the existing
  `t0_sync`/`fs_sync` scalars, which are now always a least-squares affine derived from
  the knots (or the legacy scalar, for archives predating this change). `LFPackReader`
  interpolates through the knots for `.times`/`saturation_times()` inside their range and
  falls back to the derived affine outside it (`interp1d` with no extrapolation, rather
  than the less stable slope of the two outermost knots). New `lfpack.write_sync_attrs` /
  `lfpack.clear_sync_attrs` (`src/lfpack/_sync.py`) write/clear both tiers together and
  validate the knots (>=2 points, strictly increasing, all-finite).

### Fixed
- `saturation_times()` now calls the same sample→time conversion as `.times` instead of
  duplicating the affine formula inline, so the two could never silently diverge on a
  scale with real non-linear sync structure.

## [0.3.0] - 2026-07-25

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
