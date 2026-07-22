# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Bad-channel detection labels (0=good, 1=dead, 2=noisy, 3=outside brain) are now
  persisted for quality control instead of being discarded after interpolation.
  `compress_to_h5` accepts a `labels` key in its `channels` dict (written as an int8
  attr on the scale-`00` `meta`), `compress_bin_to_h5` forwards the auto-detected
  labels, and `LFPackReader.channels` / `channels_full` expose them under the `labels`
  key (within-group mode when `bin_channels > 1`). Optional and backwards-compatible.

### Fixed
- `compress` no longer decompresses low-SNR chunks to exact zero (the WP threshold could
  wipe every coefficient of the unit-norm `Vh` rows, silently destroying real
  low-amplitude LFP). New `floor_k=64` survival floor keeps the dominant mode's largest
  coefficients; high-SNR recordings are bit-for-bit unchanged and saturation-muted spans
  still read as zero. Closes #2.

## [0.2.0] - 2026-06-30

### Added
- `LFPackReader.channels` and `LFPackReader.channels_full` — per-channel probe geometry
  and brain location annotations (`x/y/z` MNI coordinates, `atlas_id`, `acronym`).
  Brain location fields are optional; the properties work on any existing file.
  `channels` aggregates over binned-channel groups (mean for coordinates, mode for
  brain region); `channels_full` always returns the raw per-electrode data.
- `compress_to_h5` accepts an optional `channels` dict to embed brain location
  annotations at compression time.
