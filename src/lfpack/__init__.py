"""lfpack — LFP codec for Neuropixels recordings.

Lossy encode/decode pipeline: decimation → Cadzow denoising → adaptive SVD
→ wavelet-packet thresholding → HDF5 storage.
"""

from lfpack._core import (  # noqa: F401
    LFPackReader,
    LFPCompressed,
    compress,
    compress_bin_to_h5,
    compress_pipeline,
    compress_to_h5,
    decompress,
    merge_h5,
    run_cadzow_checkpoint,
    subset_h5,
)
