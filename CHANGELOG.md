# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-recording HDF5 layout: root keys are recording identifiers, scales are
  zero-padded two-digit keys (`/<recording>/<scale:02d>/meta` and `.../chunks/`).
- Pyramidal (multi-resolution) storage: multiple scale levels can coexist in the
  same file; merge via `h5py.copy`.
- `compress_to_h5` and `compress_bin_to_h5` accept a `recording=` parameter
  (defaults to the bin-file stem when omitted).
- `LFPackReader` accepts `recording=` and `scale=` keyword arguments; raises
  `ValueError` when a file contains multiple recordings and none is specified.
- `LFPackReader.recordings(h5_file)` and `LFPackReader.scales(h5_file, recording)`
  static methods for cataloguing file contents.
- Backwards-compatible fallback: legacy flat HDF5 files (with `meta` at root)
  are still readable without any changes to call sites.
- Round-trip test suite (`TestLFPackH5`, 9 tests) covering shape, fidelity,
  catalogue methods, pyramidal layout, multi-recording merge, and error paths.
- Ruff formatting and linting enforced via pre-commit hook (`.githooks/pre-commit`).

### Changed
- HDF5 files written by `compress_to_h5` now use `libver='latest'` for better
  concurrent-write safety.

## [0.1.0] - 2025-06-01

Initial release.

### Added
- `compress` / `decompress`: single-snippet SVD + wavelet-packet codec.
- `compress_pipeline`: chunked compression with Cadzow denoising.
- `compress_bin_to_h5`: streaming pipeline from a SpikeGLX `.bin` file to HDF5.
- `LFPackReader`: drop-in `spikeglx.Reader`-compatible decompressor.
- Adaptive SVD rank selection (`epsilon` threshold on noise floor).
- Per-component wavelet-packet thresholding (`alpha` multiplier).
