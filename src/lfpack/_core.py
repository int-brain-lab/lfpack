"""
LFP compression via Cadzow denoising, adaptive SVD, and wavelet-packet thresholding.

Two-stage lossy codec for local-field-potential (LFP) recordings:

  Stage 1 – Adaptive SVD (epsilon threshold)
    Rank r is selected as r = #{k : sv[k] > epsilon × sigma_noise}, where sigma_noise
    is the median of the lower half of non-trivial singular values.  This adapts the
    rank to the signal content of each snippet rather than using a fixed value.

  Stage 2 – Wavelet-packet thresholding (alpha multiplier)
    Each of the r temporal row-vectors Vh[k, :] is thresholded independently using
    tau_k = alpha × sigma_noise / sv[k], applied to db4 level-5 wavelet-packet
    coefficients.  Larger singular values (stronger spatial modes) use a lower
    threshold, preserving more of their temporal detail.

Recommended defaults (epsilon=150, alpha=28) give CR ≈ 600–1500 with
median RMSE < 5 µV on Cadzow-denoised IBL NP1/NP2 recordings.

Typical usage
-------------
>>> compressed = compress(snippet)
>>> reconstructed = decompress(compressed)

>>> reconstructed, compressed = compress_pipeline(raw_lfp, h=probe_header)
"""

from __future__ import annotations

import dataclasses
import json as _json
import os
from pathlib import Path

import neuropixel
import numpy as np
import pywt
import scipy.signal  # noqa: F401
import spikeglx as _spikeglx
from ibldsp import cadzow as _cadzow
from tqdm import tqdm

_WP_WAVELET = "db4"
_WP_MAXLEVEL = 5


@dataclasses.dataclass
class LFPCompressed:
    """Compressed representation of an (nc, ns) LFP matrix.

    Attributes
    ----------
    U_scaled : ndarray (nc, r), float32
        Left singular vectors scaled by their singular values: U[:, :r] * sv[:r].
    Vh_hat : ndarray (r, n_wp_slots), float32
        WP-domain leaf-node coefficients (alpha > 0) or time-domain rows (alpha == 0).
    ns_original : int
        Output time samples after trimming guard bands.
    epsilon : float
        SVD threshold multiplier used during compression.
    alpha : float
        WP threshold multiplier used during compression.
    cr_svd : float
        SVD-only compression ratio nc*ns / (r*(nc+ns)) — what CR would be without WP.
    cr_wp : float
        WP compression ratio r*ns / n_kept (1.0 when alpha=0).
    cr_total : float
        True compression ratio nc*ns / (r*nc + n_kept): original floats / stored floats.
    left_overlap : int
        Guard-band samples to trim at left after inverse WP (0 when no overlap context).
    ns_extended : int
        Total time samples the WP coefficients represent (0 → use ns_original).
    """

    U_scaled: np.ndarray
    Vh_hat: np.ndarray
    ns_original: int
    epsilon: float
    alpha: float
    cr_svd: float
    cr_wp: float
    cr_total: float
    left_overlap: int = 0
    ns_extended: int = 0


def _svd_noise_floor(sv: np.ndarray) -> float:
    """Median of the lower half of non-trivial singular values.

    Restricts to sv > 0.01% of max before computing the median so that
    dead/zeroed channels do not pull the noise floor to zero.
    """
    sv_nz = sv[sv > sv[0] * 1e-4]
    tail = sv_nz[sv_nz.size // 2 :] if sv_nz.size else sv
    return float(np.nanmedian(tail)) if tail.size else float(sv[0])


def _count_wp_slots(ns: int) -> int:
    """Total number of leaf wavelet-packet coefficients for a signal of length *ns*."""
    wp = pywt.WaveletPacket(data=np.zeros(ns), wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
    return sum(len(node.data) for node in wp.get_level(_WP_MAXLEVEL, "natural"))


def compress(
    data: np.ndarray,
    epsilon: float = 150.0,
    alpha: float = 28.0,
) -> LFPCompressed:
    """
    Compress an LFP snippet using adaptive SVD and wavelet-packet thresholding.

    Parameters
    ----------
    data : ndarray of shape (nc, ns)
        LFP data matrix, float32 or float64.  Rows are channels, columns are time.
    epsilon : float
        SVD threshold multiplier.  rank = #{k : sv[k] > epsilon × sigma_noise}.
        Default 150.
    alpha : float
        WP threshold multiplier per component: tau_k = alpha × sigma_noise / sv[k].
        Set to 0 to skip wavelet-packet stage.  Default 28.

    Returns
    -------
    LFPCompressed
    """
    nc, ns = data.shape
    x = data.astype(np.float64)
    U, sv, Vh = np.linalg.svd(x, full_matrices=False)

    sigma_noise = _svd_noise_floor(sv)
    r = max(1, int(np.sum(sv > epsilon * sigma_noise)))
    # cr_svd: what the CR would be with SVD alone (time-domain Vh, no WP)
    cr_svd = float(nc * ns) / (r * (nc + ns))

    n_wp_slots = _count_wp_slots(ns)
    if alpha == 0.0:
        Vh_hat = Vh[:r, :].copy()
        n_kept = r * ns  # all time-domain samples retained
    else:
        Vh_hat = np.zeros((r, n_wp_slots))
        n_kept = 0
        for k in range(r):
            tau_k = alpha * sigma_noise / (sv[k] + 1e-40)
            wp = pywt.WaveletPacket(data=Vh[k], wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
            nodes = wp.get_level(_WP_MAXLEVEL, "natural")
            offset = 0
            for node in nodes:
                mask = np.abs(node.data) >= tau_k
                n_kept += int(mask.sum())
                node_len = len(node.data)
                Vh_hat[k, offset : offset + node_len] = node.data * mask
                offset += node_len

    # cr_wp: how much WP thresholding compresses the Vh rows (1.0 when alpha=0)
    cr_wp = float(r * ns) / max(n_kept, 1)
    # cr_total: true compression ratio — original floats / (U_scaled + non-zero Vh coefficients)
    cr_total = float(nc * ns) / (r * nc + n_kept)

    return LFPCompressed(
        U_scaled=(U[:, :r] * sv[:r]).astype(np.float32),
        Vh_hat=Vh_hat.astype(np.float32),
        ns_original=ns,
        epsilon=epsilon,
        alpha=alpha,
        cr_svd=cr_svd,
        cr_wp=cr_wp,
        cr_total=cr_total,
        left_overlap=0,
        ns_extended=ns,
    )


def _reconstruct_vh_from_wp(Vh_hat_wp: np.ndarray, ns_extended: int, r: int) -> np.ndarray:
    """Inverse WP transform: flat leaf-node coefficient array → time-domain rows.

    Parameters
    ----------
    Vh_hat_wp : ndarray (r, n_wp_slots), float32
    ns_extended : int
        Signal length the WP tree was built from.
    r : int
        Number of rows.

    Returns
    -------
    ndarray (r, ns_extended), float64
    """
    wp_ref = pywt.WaveletPacket(data=np.zeros(ns_extended), wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
    node_sizes = [len(n.data) for n in wp_ref.get_level(_WP_MAXLEVEL, "natural")]

    Vh_time = np.zeros((r, ns_extended), dtype=np.float64)
    for k in range(r):
        wp = pywt.WaveletPacket(data=np.zeros(ns_extended), wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
        nodes = wp.get_level(_WP_MAXLEVEL, "natural")
        offset = 0
        for i, node in enumerate(nodes):
            sz = node_sizes[i]
            node.data = Vh_hat_wp[k, offset : offset + sz].astype(np.float64)
            offset += sz
        Vh_time[k] = wp.reconstruct(update=True)[:ns_extended]
    return Vh_time


def decompress(compressed: LFPCompressed, bin_channels: int = 1) -> np.ndarray:
    """
    Reconstruct LFP data from a compressed representation.

    Parameters
    ----------
    compressed : LFPCompressed
    bin_channels : int
        Number of adjacent channels to sum together (spatial binning).  ``1``
        means no binning.  Must evenly divide ``nc`` or trailing channels are
        silently dropped.  When > 1, the full ``(nc, ns)`` array is never
        materialised; only the binned ``(nc // bin_channels, ns)`` result is.

    Returns
    -------
    ndarray of shape (nc // bin_channels, ns_original), float32
    """
    r = compressed.U_scaled.shape[1]
    ns = compressed.ns_original
    ns_ext = compressed.ns_extended if compressed.ns_extended > 0 else ns

    lo = compressed.left_overlap
    if compressed.alpha == 0.0:
        Vh_time = compressed.Vh_hat[:, lo : lo + ns].astype(np.float64)
    else:
        Vh_time_ext = _reconstruct_vh_from_wp(compressed.Vh_hat, ns_ext, r)
        Vh_time = Vh_time_ext[:, lo : lo + ns]

    if bin_channels > 1:
        # Sum U_scaled rows in groups before the matrix multiply so the result
        # is (nc_binned, ns) rather than (nc, ns) — no large intermediate.
        nc = compressed.U_scaled.shape[0]
        nc_binned = nc // bin_channels
        U = compressed.U_scaled[: nc_binned * bin_channels].astype(np.float64)
        U_binned = U.reshape(nc_binned, bin_channels, r).sum(axis=1)
        x_hat = U_binned @ Vh_time
    else:
        x_hat = compressed.U_scaled.astype(np.float64) @ Vh_time
    return np.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compress_pipeline(
    data: np.ndarray,
    h: dict | None = None,
    epsilon: float = 150.0,
    alpha: float = 28.0,
    fs: float = 250.0,
    cadzow_rank: int = 5,
    cadzow_niter: int = 1,
    cadzow_fmax: float = 100.0,
) -> tuple[np.ndarray, LFPCompressed]:
    """
    Full LFP compression pipeline: Cadzow denoise → SVD-adapt → WP threshold.

    Parameters
    ----------
    data : ndarray of shape (nc, ns), float32
        Raw LFP data at *fs* Hz.  Rows are channels, columns are time samples.
    h : dict or None
        Neuropixel probe header containing 'x' and 'y' channel coordinates.
        Defaults to the first *nc* channels of the NP1 (version 1) geometry.
        Pass `neuropixel.trace_header(version=2)` for NP2 probes.
    epsilon : float
        SVD-adapt threshold multiplier.  Default 150.
    alpha : float
        WP threshold multiplier.  Default 28.
    fs : float
        LFP sampling rate [Hz].  Default 250.
    cadzow_rank : int
        Spatial de-rank applied by the Cadzow denoiser.  Default 5.
    cadzow_niter : int
        Number of Cadzow iterations.  Default 1.
    cadzow_fmax : float
        Maximum frequency passed to the Cadzow denoiser [Hz].  Default 100.

    Returns
    -------
    reconstructed : ndarray of shape (nc, ns), float32
        Denoised and compressed–reconstructed LFP.
    compressed : LFPCompressed
        Compression metadata and per-stage compression ratios.
    """
    nc = data.shape[0]
    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:nc] for k, v in _h.items()}

    denoised = _cadzow.cadzow_denoiser(
        data,
        h=h,
        fs=fs,
        rank=cadzow_rank,
        niter=cadzow_niter,
        fmax=cadzow_fmax,
    )
    compressed = compress(denoised, epsilon=epsilon, alpha=alpha)
    return decompress(compressed), compressed


# ── Chunk sizes for the full-recording pipeline ───────────────────────────────
# Cadzow: processed window = 768 = 3 × 256, FFT-optimal.  Halo of 64 samples
# on each side is discarded after denoising, so the written chunk is 640 samples.
# Compress: 2048 = 2^11 samples per HDF5 chunk; 128-sample guard band on each
# side covers the db4 level-5 wavelet reconstruction support (~217 samples).
_CADZOW_CHUNK = 640  # written chunk size = processed window − 2 × halo
_CADZOW_HALO = 64  # halo each side → processed = 640 + 128 = 768 = 3 × 256
_COMPRESS_CHUNK = 2048
_COMPRESS_OVERLAP = 128


def _cadzow_worker(job):
    """Process one time-chunk for run_cadzow_checkpoint (module-level for pickling)."""
    import numpy as np
    from ibldsp import cadzow as _cadzow_proc

    data = np.lib.format.open_memmap(job["data_path"], mode="r")
    out = np.lib.format.open_memmap(job["out_path"], mode="r+")

    ci, chunk, halo, ns = job["ci"], job["chunk"], job["halo"], job["ns"]
    i0_w = ci * chunk
    i1_w = min(i0_w + chunk, ns)
    i0_r = max(0, i0_w - halo)
    i1_r = min(ns, i1_w + halo)
    left_halo = i0_w - i0_r

    snippet = np.asarray(data[i0_r:i1_r, :], dtype=np.float32).T  # (nc, processed)
    denoised = _cadzow_proc.cadzow_denoiser(
        snippet,
        h=job["h"],
        fs=job["fs"],
        rank=job["rank"],
        niter=job["niter"],
        fmax=job["fmax"],
        nswx=job["nswx"],
        ovx=job["ovx"],
        gap_threshold=job["gap_threshold"],
        ppca_k=job["ppca_k"],
        n_jobs=1,
    )
    out[i0_w:i1_w, :] = denoised[:, left_halo : left_halo + (i1_w - i0_w)].T
    out.flush()
    return ci


def run_cadzow_checkpoint(
    data,
    out_npy,
    h=None,
    fs=250.0,
    chunk=_CADZOW_CHUNK,
    halo=_CADZOW_HALO,
    rank=5,
    niter=1,
    fmax=None,
    nswx=64,
    ovx=32,
    gap_threshold=2.0,
    ppca_k=2.0,
    n_jobs=4,
):
    """
    Cadzow-denoise a decimated LFP array in overlapping chunks and save a contiguous checkpoint.

    Each chunk of `chunk` samples is read with a `halo`-sample context on each side,
    denoised, then only the central `chunk` samples are written.  The output .npy
    contains no guard bands and can be memory-mapped directly.

    Parameters
    ----------
    data : ndarray (ns, nc), float32
        Decimated LFP at `fs` Hz, time-first so time slices are contiguous on disk.
    out_npy : path-like
        Output .npy file path.  Shape (ns, nc), float32, time-first.
    h : dict or None
        Probe header with keys 'x' and 'y'.  Defaults to NP1 geometry for nc channels.
    fs : float
        Sampling rate [Hz].  Default 250.
    chunk : int
        Written chunk size (samples).  chunk + 2*halo must be FFT-optimal.
        Default 640 → processed window = 640 + 2×64 = 768 = 3 × 256.
    halo : int
        Context halo each side (samples).  Default 64.
    rank : int
        Cadzow SVD rank.  Default 5.
    niter : int
        Number of Cadzow iterations.  Default 1.
    fmax : float or None
        Max frequency for Cadzow [Hz].  None → Nyquist.  Default None.
    nswx : int
        Cadzow channel-window width.  Default 64.
    ovx : int
        Cadzow channel-window overlap.  Default 32 (50% of nswx).
    gap_threshold : float
        Adaptive-rank gap threshold.  Default 2.0.
    ppca_k : float
        PPCA outlier-suppression threshold.  Default 2.0.
    n_jobs : int
        Number of chunks processed in parallel via ProcessPoolExecutor.  Each
        worker subprocess calls cadzow_denoiser with n_jobs=1 and writes its
        result directly to the output memmap.  Default 4.

    Returns
    -------
    ndarray (ns, nc), float32 — also written to out_npy.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ns, nc = data.shape
    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:nc] for k, v in _h.items()}

    out_npy = Path(out_npy)

    # Create the output .npy file up-front so workers can open it with mode='r+'
    out_mm = np.lib.format.open_memmap(str(out_npy), mode="w+", dtype=np.float32, shape=(ns, nc))
    del out_mm  # flush header + allocation; workers re-open independently

    # Workers need a file path, not an array.  Memmaps expose .filename; otherwise
    # save a temp file so every subprocess can re-open the data without IPC copies.
    _tmp_input = None
    if hasattr(data, "filename"):
        data_path = str(data.filename)
    else:
        _tmp_input = out_npy.with_suffix(".tmp_input.npy")
        np.save(_tmp_input, data)
        data_path = str(_tmp_input)

    n_chunks = int(np.ceil(ns / chunk))
    shared = dict(
        data_path=data_path,
        out_path=str(out_npy),
        ns=ns,
        chunk=chunk,
        halo=halo,
        h=h,
        fs=fs,
        rank=rank,
        niter=niter,
        fmax=fmax,
        nswx=nswx,
        ovx=ovx,
        gap_threshold=gap_threshold,
        ppca_k=ppca_k,
    )
    jobs = [{**shared, "ci": ci} for ci in range(n_chunks)]

    from tqdm import tqdm

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_cadzow_worker, job): job["ci"] for job in jobs}
        with tqdm(total=n_chunks, desc="Cadzow", unit="chunk") as pbar:
            for fut in as_completed(futures):
                fut.result()  # re-raise any worker exception immediately
                pbar.update(1)

    if _tmp_input is not None:
        _tmp_input.unlink()

    return np.load(out_npy, mmap_mode="r")


def _compress_chunk_worker(args):
    """Compress one chunk (module-level for pickling by ProcessPoolExecutor)."""
    npy_path, i0_r, i1_r, n_w, left_ov, epsilon, alpha = args
    data = np.load(npy_path, mmap_mode="r")
    snippet = np.asarray(data[i0_r:i1_r, :], dtype=np.float32).T
    c = compress(snippet, epsilon=epsilon, alpha=alpha)
    reconstructed = decompress(c)
    rmse = float(
        np.sqrt(
            np.mean(
                (
                    snippet[:, left_ov : left_ov + n_w].astype(np.float64)
                    - reconstructed[:, left_ov : left_ov + n_w].astype(np.float64)
                )
                ** 2
            )
        )
    )
    flat = c.Vh_hat.ravel()
    vh_idx = np.flatnonzero(flat).astype(np.int32)
    return {
        "U_scaled": c.U_scaled,
        "vh_indices": vh_idx,
        "vh_values": flat[vh_idx],
        "vh_shape": c.Vh_hat.shape,
        "ns_original": n_w,
        "ns_extended": snippet.shape[1],
        "left_overlap": left_ov,
        "epsilon": epsilon,
        "alpha": alpha,
        "cr_svd": c.cr_svd,
        "cr_wp": c.cr_wp,
        "cr_total": c.cr_total,
        "rmse": rmse,
    }


def compress_to_h5(
    cadzow_npy,
    out_h5,
    recording,
    scale=0,
    sglx_meta=None,
    h=None,
    chunk=_COMPRESS_CHUNK,
    overlap=_COMPRESS_OVERLAP,
    epsilon=150.0,
    alpha=28.0,
    fs=250.0,
    n_jobs=4,
):
    """
    Compress a Cadzow-denoised .npy into a single HDF5 archive of LFPCompressed chunks.

    Each written chunk of `chunk` samples is extended by `overlap` samples on each side
    before SVD + wavelet-packet compression.  Only the central `chunk` columns of Vh_hat
    are stored, eliminating wavelet-reconstruction boundary artefacts.  Decompressed
    chunks are concatenated without overlap during reading.

    HDF5 layout
    -----------
    /<recording>/<scale_str>/meta        attrs: nc, ns_total, fs, compress_chunk,
                                                compress_overlap, epsilon, alpha,
                                                sglx_meta (JSON), geometry_x, geometry_y
    /<recording>/<scale_str>/chunks/<i>/ datasets: U_scaled (nc, r),
                                                   vh_indices (n_kept,) int32,
                                                   vh_values (n_kept,) float32
                                         attrs: ns_original, ns_extended, left_overlap,
                                                vh_shape, epsilon, alpha, cr_svd, cr_wp,
                                                cr_total, rmse

    where <scale_str> = f'{scale:02d}', e.g. '00', '01', …  Multiple recordings and/or
    scales can coexist in a single file; merging two files is a plain group copy.

    Parameters
    ----------
    cadzow_npy : path-like
        Path to the (ns, nc) float32 Cadzow checkpoint (time-first).
    out_h5 : path-like
        Output HDF5 file (created or overwritten).
    recording : str
        Unique key for this recording (e.g. a probe-insertion UUID).  Top-level HDF5
        group name; allows multiple recordings to coexist in one file.
    scale : int
        Resolution level (zero-padded to two digits in the path).  0 = base resolution.
        Default 0.
    sglx_meta : dict or None
        Original spikeglx metadata (sr.meta).  Stored verbatim as JSON.
    h : dict or None
        Probe header.  Defaults to NP1 geometry for nc channels.
    chunk : int
        Written chunk size (samples).  Default 2048 = 2^11.
    overlap : int
        Guard-band samples each side.  Default 128.
    epsilon : float
        SVD threshold multiplier.  Default 150.
    alpha : float
        WP threshold multiplier.  Default 28.
    fs : float
        Sampling rate [Hz] written into metadata.  Default 250.
    """
    import h5py

    data = np.load(cadzow_npy, mmap_mode="r")  # (ns, nc) time-first
    ns, nc = data.shape
    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:nc] for k, v in _h.items()}

    out_h5 = Path(out_h5)
    n_chunks = int(np.ceil(ns / chunk))
    total_cr = 0.0

    from tqdm import tqdm

    jobs = []
    for ci in range(n_chunks):
        i0_w = ci * chunk
        i1_w = min(i0_w + chunk, ns)
        n_w = i1_w - i0_w
        i0_r = max(0, i0_w - overlap)
        i1_r = min(ns, i1_w + overlap)
        jobs.append((str(cadzow_npy), i0_r, i1_r, n_w, i0_w - i0_r, epsilon, alpha))

    root = f"{recording}/{scale:02d}"
    with h5py.File(out_h5, "w", libver="latest") as f:
        mg = f.create_group(f"{root}/meta")
        mg.attrs["nc"] = nc
        mg.attrs["ns_total"] = ns
        mg.attrs["fs"] = fs
        mg.attrs["compress_chunk"] = chunk
        mg.attrs["compress_overlap"] = overlap
        mg.attrs["epsilon"] = epsilon
        mg.attrs["alpha"] = alpha
        mg.attrs["sglx_meta"] = _json.dumps(sglx_meta or {})
        mg.attrs["geometry_x"] = h["x"].astype(np.float32)
        mg.attrs["geometry_y"] = h["y"].astype(np.float32)

        cg = f.create_group(f"{root}/chunks")
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compress_chunk_worker)(job) for job in tqdm(jobs, desc="Compress", unit="chunk")
        )
        for ci, r in enumerate(results):
            grp = cg.create_group(str(ci))
            grp.create_dataset("U_scaled", data=r["U_scaled"], compression="gzip", shuffle=True)
            grp.create_dataset("vh_indices", data=r["vh_indices"], compression="gzip", shuffle=True)
            grp.create_dataset("vh_values", data=r["vh_values"], compression="gzip", shuffle=True)
            grp.attrs["vh_shape"] = r["vh_shape"]
            grp.attrs["ns_original"] = r["ns_original"]
            grp.attrs["ns_extended"] = r["ns_extended"]
            grp.attrs["left_overlap"] = r["left_overlap"]
            grp.attrs["epsilon"] = r["epsilon"]
            grp.attrs["alpha"] = r["alpha"]
            grp.attrs["cr_svd"] = r["cr_svd"]
            grp.attrs["cr_wp"] = r["cr_wp"]
            grp.attrs["cr_total"] = r["cr_total"]
            grp.attrs["rmse"] = r["rmse"]
            total_cr += r["cr_total"]

    print(f"Saved {out_h5}  mean CR={total_cr / n_chunks:.0f}")
    return out_h5


def merge_h5(src_files, dst_h5, recording_map=None):
    """
    Merge multiple per-recording HDF5 files into one multi-recording archive.

    Each source file must contain exactly one top-level recording group (the
    layout produced by ``compress_to_h5`` and ``compress_bin_to_h5``).  The
    entire group is copied verbatim — no re-compression is performed.

    Parameters
    ----------
    src_files : sequence of path-like
        Source HDF5 files, one recording per file.
    dst_h5 : path-like
        Output multi-recording HDF5 file (always created fresh).
    recording_map : dict mapping path-like to str, optional
        Override the recording name for specific source files.  Keys are
        matched by resolved absolute path.  Files absent from the map retain
        their original top-level group name.

    Returns
    -------
    Path
        Resolved path to *dst_h5*.

    Raises
    ------
    ValueError
        If a source file contains more than one top-level group, or if two
        source files would resolve to the same recording name.
    """
    import h5py

    resolved_map = {Path(k).resolve(): v for k, v in (recording_map or {}).items()}

    # Build the plan before touching dst_h5 so errors surface early.
    plan = []
    for src in src_files:
        src_path = Path(src).resolve()
        with h5py.File(src_path, "r") as f:
            keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{src_path.name} has {len(keys)} top-level groups {keys}; "
                "merge_h5 requires exactly one recording per source file"
            )
        recording = resolved_map.get(src_path, keys[0])
        plan.append((src_path, recording, keys[0]))

    seen: set = set()
    dupes: list = []
    for _, recording, _ in plan:
        if recording in seen:
            dupes.append(recording)
        else:
            seen.add(recording)
    if dupes:
        raise ValueError(f"Duplicate recording name(s): {sorted(set(dupes))}")

    dst_h5 = Path(dst_h5)
    with h5py.File(dst_h5, "w") as dst:
        for src_path, recording, src_key in tqdm(plan, desc=dst_h5.stem, unit="PID"):
            with h5py.File(src_path, "r") as src:
                src.copy(src_key, dst, name=recording)

    return dst_h5.resolve()


def compress_bin_to_h5(
    bin_file,
    out_h5,
    recording=None,
    q=10,
    h=None,
    cadzow_checkpoint_file=None,
    cadzow_kwargs=None,
    channel_labels=None,
    epsilon=150.0,
    alpha=28.0,
    n_jobs=4,
    chunk=_COMPRESS_CHUNK,
    overlap=_COMPRESS_OVERLAP,
    highpass_cutoff=2.0,
    car=True,
    fig_dir=None,
):
    """
    Full pipeline: raw LFP binary → decimate → Cadzow denoise → SVD+WP compress → HDF5.

    Decimation uses ibldsp.voltage.resample_denoise_lfp_cbin (FIR anti-aliasing).  Cadzow
    denoising is performed inside each decimation worker when *cadzow_kwargs* is provided.
    An intermediate float32 checkpoint (.npy) is always written — either to the path given
    by *cadzow_checkpoint_file* or to a sibling temp file that is deleted after the HDF5 is
    finalised.  If the checkpoint file already exists its contents are used directly, skipping
    the expensive decimate+denoise step.

    Bad channels are detected automatically (via ibldsp.voltage.detect_bad_channels_cbin)
    before decimation unless *channel_labels* is supplied or the checkpoint already exists.
    Detected bad channels are interpolated by resample_denoise_lfp_cbin before SVD, which
    prevents incoherent channels from collapsing the noise-floor estimate and inflating rank.

    Parameters
    ----------
    bin_file : path-like
        SpikeGLX LFP binary (.cbin or .bin).  The .meta file must be in the same directory.
    out_h5 : path-like
        Output HDF5 file (created or overwritten).
    recording : str or None
        Unique key for this recording (e.g. a probe-insertion UUID).  Stored as the
        top-level HDF5 group; multiple recordings can coexist in one file.
        Defaults to the stem of bin_file when None.
    q : int
        Decimation factor.  Default 10 (2500 → 250 Hz).
    h : dict or None
        Probe header with keys 'x' and 'y'.  Defaults to NP1 geometry for nc channels.
    cadzow_checkpoint_file : path-like or None
        Path for the intermediate Cadzow .npy checkpoint (ns_lf, nc) float32.
        If None a temporary file is written next to out_h5 and deleted afterwards.
        If the file already exists the decimate+Cadzow step is skipped entirely.
    cadzow_kwargs : dict or None
        Forwarded to resample_denoise_lfp_cbin as cadzow_kwargs; keys match
        ibldsp.cadzow.cadzow_denoiser parameters (rank, niter, fmax, nswx, ovx,
        gap_threshold, ppca_k).  Default None disables Cadzow (pure decimation).
    channel_labels : np.ndarray or None
        Per-channel quality labels (0=good, 1=dead, 2=noisy, 3=outside brain).
        If None and the checkpoint does not exist, labels are auto-detected via
        ibldsp.voltage.detect_bad_channels_cbin.  Pass an array of zeros to skip
        detection explicitly.
    epsilon : float
        SVD threshold multiplier.  Default 150.
    alpha : float
        WP threshold multiplier.  Default 28.
    n_jobs : int
        Parallel workers for the decimate+Cadzow stage.  Default 4.
    chunk : int
        Compress chunk size in decimated samples.  Default 2048.
    overlap : int
        SVD guard-band samples each side.  Default 128.
    highpass_cutoff : float or None
        3rd-order Butterworth zero-phase highpass corner [Hz] applied before decimation.
        Default 2.0 Hz.  None disables the filter.
    car : bool
        Apply median common-average reference before decimation.  Default True.
    fig_dir : path-like or None
        If set, a bad-channel diagnostic figure is saved to this directory after detection.
        Uses ibldsp.plots.show_channels_labels on a single mid-recording batch.
        Filename: ``bad_channels_{bin_file.stem}.png``.  Default None (no figure).

    Returns
    -------
    Path
        Path to the output HDF5 file.
    """
    from ibldsp.voltage import detect_bad_channels_cbin, resample_denoise_lfp_cbin

    bin_file = Path(bin_file)
    out_h5 = Path(out_h5)
    if recording is None:
        recording = bin_file.stem
    n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

    sr = _spikeglx.Reader(bin_file)
    nc = sr.nc - sr.nsync
    fs_lf = sr.fs / q
    sglx_meta = sr.meta

    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:nc] for k, v in _h.items()}

    # Determine checkpoint path
    if cadzow_checkpoint_file is None:
        cadzow_npy = out_h5.with_suffix(".cadzow_tmp.npy")
        delete_checkpoint = True
    else:
        cadzow_npy = Path(cadzow_checkpoint_file)
        delete_checkpoint = False

    # Stage 1: decimate (+ optional Cadzow) → float32 checkpoint
    if cadzow_npy.exists():
        print(f"Using existing Cadzow checkpoint {cadzow_npy}")
    else:
        if channel_labels is None:
            print("Detecting bad channels …")
            channel_labels, xfeats_med = detect_bad_channels_cbin(sr, return_features=True)
            n_bad = int(np.sum(channel_labels != 0))
            print(f"  {n_bad} / {nc} channels flagged (labels: {np.unique(channel_labels, return_counts=True)})")
            if fig_dir is not None:
                import matplotlib.pyplot as plt
                from ibldsp.plots import show_channels_labels

                batch_dur = 1e4 / sr.fs
                t_mid = (sr.rl - batch_dur) / 2
                sl = slice(int(t_mid * sr.fs), int((t_mid + batch_dur) * sr.fs))
                raw_batch = sr[sl, :nc].T
                fig, _ = show_channels_labels(raw_batch, sr.fs, channel_labels, xfeats_med, psd_hf_threshold=1.4)
                fig.suptitle(bin_file.stem, fontsize=9)
                fig_path = Path(fig_dir).joinpath(f"bad_channels_{bin_file.stem}.png")
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
                print(f"  Channel labels figure → {fig_path}")
        resample_denoise_lfp_cbin(
            bin_file,
            q=q,
            output=cadzow_npy,
            dtype=np.float32,
            channel_labels=channel_labels,
            highpass_cutoff=highpass_cutoff,
            car=car,
            cadzow_kwargs=cadzow_kwargs,
            n_jobs=n_jobs,
        )

    # Stage 2: compress checkpoint → HDF5
    compress_to_h5(
        cadzow_npy,
        out_h5,
        recording=recording,
        sglx_meta=sglx_meta,
        h=h,
        chunk=chunk,
        overlap=overlap,
        epsilon=epsilon,
        alpha=alpha,
        fs=fs_lf,
        n_jobs=n_jobs,
    )

    if delete_checkpoint:
        cadzow_npy.unlink()

    return out_h5


class LFPackReader(_spikeglx.Reader):
    """
    Drop-in spikeglx.Reader for HDF5-packed LFP-compressed files.

    Chunks are decompressed on demand.  No sync trace is available; read() with
    sync=True returns None as the second element.  Data is returned in volts (float32)
    in the same (n_samples, n_channels) convention as spikeglx.Reader.

    The HDF5 layout is /<recording>/<scale_str>/meta  and  /<recording>/<scale_str>/chunks/.
    A file may contain multiple recordings and/or multiple scale levels.  When a file
    contains exactly one recording the key is auto-detected; otherwise pass recording=.

    Parameters
    ----------
    h5_file : path-like
        HDF5 archive produced by compress_to_h5.
    recording : str or None
        Recording key (top-level group name).  Auto-detected when the file contains
        exactly one recording; raises ValueError for multi-recording files.
    scale : int
        Resolution level to open.  0 = base (full LFP rate).  Default 0.
    bin_channels : int
        Number of adjacent channels to sum together on every read.  ``1``
        (default) means no binning.  When set, ``nc``, ``shape``, and
        ``geometry`` all reflect the binned dimension, and slicing
        (``sr[0:2500, :]``) returns ``(n_samples, nc // bin_channels)``
        without any extra arguments.

    Examples
    --------
    >>> sr = LFPackReader('lf_compressed.h5')
    >>> sr[0:2500, :]                              # (2500, nc)
    >>> sr4 = LFPackReader('lf_compressed.h5', bin_channels=4)
    >>> sr4[0:2500, :]                             # (2500, nc // 4)
    >>> sr4.nc                                     # nc // 4
    >>> sr4.shape                                  # (ns, nc // 4)
    >>> sr4.geometry['y'].shape                    # (nc // 4,)
    """

    def __init__(self, h5_file, recording=None, scale=0, bin_channels=1):
        import h5py

        self._h5_file = Path(h5_file)
        self._h5 = None
        self._raw = None  # is_open sentinel (None → closed)
        self._geometry = None
        self.ignore_warnings = False
        self.file_bin = self._h5_file
        self.file_meta_data = None
        self.meta = None  # None → base-class properties fall back to _nc/_fs/_ns
        self.dtype = np.dtype("float32")
        self.ch_file = None
        self._bin_channels = bin_channels

        with h5py.File(self._h5_file, "r") as f:
            if "meta" in f:  # legacy single-recording format (no recording/scale hierarchy)
                self._root = None
            else:
                root_keys = list(f.keys())
                if recording is None:
                    if len(root_keys) == 1:
                        recording = root_keys[0]
                    else:
                        raise ValueError(f"Multiple recordings in file, specify recording= from: {root_keys}")
                elif recording not in f:
                    raise KeyError(f"Recording '{recording}' not found. Available: {root_keys}")
                self._root = f"{recording}/{scale:02d}"
            meta_path = f"{self._root}/meta" if self._root else "meta"
            chunks_path = f"{self._root}/chunks" if self._root else "chunks"
            attrs = f[meta_path].attrs
            self._nc = int(attrs["nc"])
            self._ns = int(attrs["ns_total"])
            self._fs = float(attrs["fs"])
            self._compress_chunk = int(attrs["compress_chunk"])
            self._n_chunks = len(f[chunks_path])
            self.sglx_meta = _json.loads(attrs["sglx_meta"])
            self._geometry = {
                "x": attrs["geometry_x"][:].astype(np.float32),
                "y": attrs["geometry_y"][:].astype(np.float32),
            }

        self._nsync = 0
        # Data is already in volts; s2v = 1.0 for all channels.
        self.channel_conversion_sample2v = {"samples": np.ones(self._nc, dtype=np.float32)}
        self.open()

    @property
    def bin_channels(self):
        """Number of adjacent channels summed on every read (1 = no binning)."""
        return self._bin_channels

    @bin_channels.setter
    def bin_channels(self, value):
        self._bin_channels = int(value)

    @property
    def nc(self):
        """Number of output channels (raw nc // bin_channels)."""
        return self._nc // self._bin_channels

    @property
    def geometry(self):
        """Probe geometry averaged over each bin group.

        When ``bin_channels == 1`` this is identical to ``geometry_full``.
        Use ``geometry_full`` to always get the raw per-electrode positions.

        Returns
        -------
        dict with keys 'x' and 'y', each an ndarray of shape (nc,).
        """
        if self._bin_channels == 1:
            return self._geometry
        n = self._bin_channels
        nc_binned = self._nc // n
        return {k: self._geometry[k][: nc_binned * n].reshape(nc_binned, n).mean(axis=1) for k in ("x", "y")}

    @geometry.setter
    def geometry(self, value):
        # spikeglx.Reader base class assigns self.geometry = None in some paths;
        # route those writes to the private backing store.
        self._geometry = value

    @property
    def geometry_full(self):
        """Full per-electrode probe geometry, independent of ``bin_channels``.

        Includes a ``'binned_channel_index'`` field mapping each raw channel to its
        corresponding output channel index (``raw_channel // bin_channels``).

        Returns
        -------
        dict with keys 'x', 'y', and 'binned_channel_index', each an ndarray of shape (nc_raw,).
        """
        n = self._bin_channels
        binned_channel_index = np.arange(self._nc, dtype=np.int32) // n
        return {**self._geometry, "binned_channel_index": binned_channel_index}

    @staticmethod
    def recordings(h5_file):
        """List recording keys at the root of an H5 file written by compress_to_h5.

        Parameters
        ----------
        h5_file : path-like

        Returns
        -------
        list of str
        """
        import h5py

        with h5py.File(h5_file, "r") as f:
            if "meta" in f:  # legacy format
                return []
            return list(f.keys())

    @staticmethod
    def scales(h5_file, recording):
        """List scale indices available for a recording.

        Parameters
        ----------
        h5_file : path-like
        recording : str

        Returns
        -------
        list of int
        """
        import h5py

        with h5py.File(h5_file, "r") as f:
            if recording not in f:
                raise KeyError(f"Recording '{recording}' not found")
            return sorted(int(k) for k in f[recording].keys())

    def open(self):
        import h5py

        self._h5 = h5py.File(self._h5_file, "r")
        self._raw = True  # non-None sentinel so base-class is_open returns True

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._raw = None

    @property
    def is_open(self):
        return self._h5 is not None

    @property
    def is_mtscomp(self):
        return False

    @property
    def fs(self):
        return self._fs

    @property
    def ns(self):
        return self._ns

    def read(self, nsel=slice(0, 10000), csel=slice(None), sync=True, bin_channels=None):
        """
        Decompress and return a sample range.

        Parameters
        ----------
        nsel : slice or int
            Sample selection (Python slice convention).
        csel : slice or array-like
            Channel selection applied after spatial binning.
        sync : bool
            If True returns (data, None); no sync trace in compressed files.
        bin_channels : int or None
            Number of adjacent channels to sum together.  ``None`` uses
            ``self.bin_channels``.  ``csel`` indexes into the binned channels.

        Returns
        -------
        data : ndarray (n_samples, nc // bin_channels), float32, volts
        sync : None  (only when sync=True)
        """
        if bin_channels is None:
            bin_channels = self._bin_channels
        if not self.is_open:
            raise IOError("Reader not open; call open() first.")

        if isinstance(nsel, int):
            first_sample, last_sample = nsel, nsel + 1
        else:
            first_sample = nsel.start if nsel.start is not None else 0
            last_sample = nsel.stop if nsel.stop is not None else self._ns
        first_sample = max(0, first_sample)
        last_sample = min(self._ns, last_sample)

        chunk = self._compress_chunk
        first_chunk = first_sample // chunk
        last_chunk = (last_sample - 1) // chunk

        pieces = []
        for ci in range(first_chunk, last_chunk + 1):
            chunk_path = f"{self._root}/chunks/{ci}" if self._root else f"chunks/{ci}"
            grp = self._h5[chunk_path]
            ns_orig = int(grp.attrs["ns_original"])
            # Reconstruct dense Vh_hat from sparse storage
            vh_shape = tuple(int(x) for x in grp.attrs["vh_shape"])
            Vh_hat = np.zeros(vh_shape, dtype=np.float32)
            Vh_hat.ravel()[grp["vh_indices"][:]] = grp["vh_values"][:]
            c = LFPCompressed(
                U_scaled=grp["U_scaled"][:],
                Vh_hat=Vh_hat,
                ns_original=ns_orig,
                epsilon=float(grp.attrs["epsilon"]),
                alpha=float(grp.attrs["alpha"]),
                cr_svd=float(grp.attrs["cr_svd"]),
                cr_wp=float(grp.attrs["cr_wp"]),
                cr_total=float(grp.attrs["cr_total"]),
                left_overlap=int(grp.attrs.get("left_overlap", 0)),
                ns_extended=int(grp.attrs.get("ns_extended", ns_orig)),
            )
            pieces.append(decompress(c, bin_channels=bin_channels))  # (nc[_binned], ns_chunk_i)

        full = np.concatenate(pieces, axis=1)  # (nc[_binned], total_samples)
        start = first_sample - first_chunk * chunk
        data = full[:, start : start + (last_sample - first_sample)]  # (nc[_binned], n_req)

        # Transpose to spikeglx convention (n_samples, nc[_binned])
        data = data.T.astype(np.float32)
        if not (isinstance(csel, slice) and csel == slice(None)):
            data = data[:, csel]

        if sync:
            return data, None
        return data

    def read_samples(self, first_sample=0, last_sample=10000, channels=None, bin_channels=None):
        """
        Read and decompress a sample range with optional spatial binning.

        Parameters
        ----------
        first_sample : int
        last_sample : int
        channels : slice or array-like or None
            Channel selection applied after binning.  ``None`` selects all.
        bin_channels : int or None
            Number of adjacent channels to sum together.  ``None`` uses
            ``self.bin_channels``.  Valid values: 1, 2, 4, 6, 8, 12.

        Returns
        -------
        ndarray (n_samples, nc // bin_channels), float32, volts
        """
        if channels is None:
            channels = slice(None)
        return self.read(slice(first_sample, last_sample), channels, bin_channels=bin_channels)
