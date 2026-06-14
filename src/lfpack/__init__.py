"""lfpack — LFP codec for Neuropixels recordings.

Lossy encode/decode pipeline: decimation → Cadzow denoising → adaptive SVD
→ wavelet-packet thresholding → HDF5 storage.
"""
from lfpack._core import (  # noqa: F401
    LFPCompressed,
    LFPCompressedReader,
    compress,
    compress_pipeline,
    compress_to_h5,
    decompress,
    run_cadzow_checkpoint,
)