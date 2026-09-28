# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Changed
- **HDF5 format 2** (breaking: lfpack only writes format 2). The codec chunks of a scale are
  stored in one flat `codec/` group (`chunk_table`, `basis`, `u_values`, `u_norms`,
  `vh_deltas`, `vh_values`) instead of one HDF5 group per 8 s chunk, with a **shared spatial
  basis** per recording and scale: one (nc, 32) float32 basis plus float16 per-chunk
  coefficients replace the float32 `U_scaled` of every chunk. Vh indices are delta-encoded
  uint16 and the coefficients that only reconstruct the guard bands are dropped. Random
  access is unchanged; on a 1.4 h NP1 recording at ε = 100, α = 14 the file goes from
  38.7 MB (format 1) to 15.9 MB for < 0.01 dB of SNR.
- Default codec parameters are now `epsilon=100`, `alpha=7` (were 150 and 28): at the same
  file size as the former 0.4 "mild" tier, behaviour decoding moves closer to uncompressed.
  `alpha` is the parameter that sets decoding quality; 14 and 2.5 bracket it.

### Added
- `basis_size` parameter (default 32) on `compress_to_h5` and `compress_bin_to_h5`.
- `lfpack.upgrade_h5(src, dst)` converts a format-1 archive to format 2 without
  re-compressing.

### Deprecated
- Format 1 (`chunks/<i>/` groups) is read-only and will be removed in a future release;
  its support is isolated in `lfpack._legacy`.

### Fixed
- `LFPackReader` accepts an already-open binary file-like object (e.g. an `s3fs`
  handle) as `h5_file`, not just a path — enables reading an archive directly off
  remote storage without a local mirror.

## [0.4.0] - 2026-08-23

### Added
- Non-linear sync support: `sync_samples`/`sync_times` — the raw ALF sync knot pairs
  (sample index <-> time), stored verbatim per scale — alongside the existing
  `t0_sync`/`fs_sync` scalars, which are now always a least-squares affine derived from
  the knots (or the legacy scalar, for archives predating this change). `LFPackReader`
  interpolates through the knots for `.times`/`saturation_times()` inside their range and
  falls back to the derived affine outside it (`interp1d` with no extrapolation, rather
  than the less stable slope of the two outermost knots). New `lfpack.write_sync_attrs` /
  `lfpack.clear_sync_attrs` (`src/lfpack/_sync.py`) write/clear both tiers together and
  validate the knots (>=2 points, strictly increasing, all-finite). `sync_samples`/
  `sync_times` are stored as gzip+shuffle child datasets of `meta`, not attrs — a
  `type='exact'` fit's knots (every raw pulse, verbatim) can run to tens of thousands of
  points, well past HDF5's per-attribute object-header-message size limit.
- Added `docs/how-to/write-your-own-file.qmd`, documenting the two sync styles (linear
  affine vs. piecewise knots) and moving/fixing the multi-recording write example that
  used to live in `how-to/multi-recording.qmd`.

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
