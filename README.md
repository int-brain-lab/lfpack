# lfpack — LFP codec for Neuropixels recordings

<p align="center">
  <img src="docs/figures/logo.png" alt="lfpack logo" width="300"/>
</p>

Lossy codec for local-field-potential (LFP) recordings from Neuropixels probes.
Achieves **>100× compression** with median RMSE < 25 µV via an 8-stage pipeline
(bad-channel detection → dephasing → highpass → interpolation → CAR → decimation → Cadzow → adaptive SVD + wavelet-packet thresholding).

> **IBL Brain-Wide Map LFP dataset** — 699 recordings, 384 channels, session-clock aligned.
> [How to access →](https://int-brain-lab.github.io/lfpack/how-to/bwm-dataset.html)

```bash
pip install lfpack
```

## Visualise with viewephys

[`viewephys`](https://github.com/int-brain-lab/viewephys) opens lfpack `.h5` files natively —
`viewephys -f recording.h5` gets you a browsable, brain-region-coloured view with no
manual decompression step:

<p align="center">
  <img src="docs/figures/viewephys_screenshot.jpg" alt="viewephys opened directly on a lfpack HDF5 file, showing brain-region-coloured LFP traces" width="700"/>
</p>

See the [BWM how-to](https://int-brain-lab.github.io/lfpack/how-to/bwm-dataset.html#visualise-with-viewephys) for details.

## Documentation

Full documentation is at **https://int-brain-lab.github.io/lfpack/**.

| Section | Contents |
| --- | --- |
| [Tutorial](https://int-brain-lab.github.io/lfpack/tutorials/first-compression.html) | End-to-end compression and decompression of a recording |
| [How-To: binned reads](https://int-brain-lab.github.io/lfpack/how-to/binned-reads.html) | Memory-efficient channel-binned access |
| [How-To: multi-recording files](https://int-brain-lab.github.io/lfpack/how-to/multi-recording.html) | Combining multiple recordings in one HDF5 file |
| [API reference](https://int-brain-lab.github.io/lfpack/reference/) | Full public API (`compress_bin_to_h5`, `LFPackReader`, …) |
| [HDF5 format](https://int-brain-lab.github.io/lfpack/reference/hdf5-layout.html) | On-disk layout specification |
| [Pipeline explanation](https://int-brain-lab.github.io/lfpack/explanation/pipeline.html) | Stage-by-stage description of the compression pipeline |
| [SVD+WP benchmark](https://int-brain-lab.github.io/lfpack/explanation/benchmark.html) | RMSE, SNR, and compression-ratio results across 11 insertions |