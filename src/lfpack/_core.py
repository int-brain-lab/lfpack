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
from pathlib import Path

import numpy as np
import pywt
import scipy.signal  # noqa: F401

import neuropixel
import spikeglx as _spikeglx
from ibldsp import cadzow as _cadzow

_WP_WAVELET = 'db4'
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
    tail = sv_nz[sv_nz.size // 2:] if sv_nz.size else sv
    return float(np.nanmedian(tail)) if tail.size else float(sv[0])


def _count_wp_slots(ns: int) -> int:
    """Total number of leaf wavelet-packet coefficients for a signal of length *ns*."""
    wp = pywt.WaveletPacket(data=np.zeros(ns), wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
    return sum(len(node.data) for node in wp.get_level(_WP_MAXLEVEL, 'natural'))


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
        n_kept = r * ns   # all time-domain samples retained
    else:
        Vh_hat = np.zeros((r, n_wp_slots))
        n_kept = 0
        for k in range(r):
            tau_k = alpha * sigma_noise / (sv[k] + 1e-40)
            wp = pywt.WaveletPacket(data=Vh[k], wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
            nodes = wp.get_level(_WP_MAXLEVEL, 'natural')
            offset = 0
            for node in nodes:
                mask = np.abs(node.data) >= tau_k
                n_kept += int(mask.sum())
                node_len = len(node.data)
                Vh_hat[k, offset:offset + node_len] = node.data * mask
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
    node_sizes = [len(n.data) for n in wp_ref.get_level(_WP_MAXLEVEL, 'natural')]

    Vh_time = np.zeros((r, ns_extended), dtype=np.float64)
    for k in range(r):
        wp = pywt.WaveletPacket(data=np.zeros(ns_extended), wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
        nodes = wp.get_level(_WP_MAXLEVEL, 'natural')
        offset = 0
        for i, node in enumerate(nodes):
            sz = node_sizes[i]
            node.data = Vh_hat_wp[k, offset:offset + sz].astype(np.float64)
            offset += sz
        Vh_time[k] = wp.reconstruct(update=True)[:ns_extended]
    return Vh_time


def decompress(compressed: LFPCompressed) -> np.ndarray:
    """
    Reconstruct LFP data from a compressed representation.

    Parameters
    ----------
    compressed : LFPCompressed

    Returns
    -------
    ndarray of shape (nc, ns_original), float32
    """
    r = compressed.U_scaled.shape[1]
    ns = compressed.ns_original
    ns_ext = compressed.ns_extended if compressed.ns_extended > 0 else ns

    lo = compressed.left_overlap
    if compressed.alpha == 0.0:
        Vh_time = compressed.Vh_hat[:, lo:lo + ns].astype(np.float64)
    else:
        Vh_time_ext = _reconstruct_vh_from_wp(compressed.Vh_hat, ns_ext, r)
        Vh_time = Vh_time_ext[:, lo:lo + ns]

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
        data, h=h, fs=fs, rank=cadzow_rank, niter=cadzow_niter, fmax=cadzow_fmax,
    )
    compressed = compress(denoised, epsilon=epsilon, alpha=alpha)
    return decompress(compressed), compressed


# ── Chunk sizes for the full-recording pipeline ───────────────────────────────
# Cadzow: processed window = 768 = 3 × 256, FFT-optimal.  Halo of 64 samples
# on each side is discarded after denoising, so the written chunk is 640 samples.
# Compress: 2048 = 2^11 samples per HDF5 chunk; 128-sample guard band on each
# side covers the db4 level-5 wavelet reconstruction support (~217 samples).
_CADZOW_CHUNK = 640    # written chunk size = processed window − 2 × halo
_CADZOW_HALO = 64      # halo each side → processed = 640 + 128 = 768 = 3 × 256
_COMPRESS_CHUNK = 2048
_COMPRESS_OVERLAP = 128


def _cadzow_worker(job):
    """Process one time-chunk for run_cadzow_checkpoint (module-level for pickling)."""
    import numpy as np
    from ibldsp import cadzow as _cadzow_proc

    data = np.lib.format.open_memmap(job['data_path'], mode='r')
    out = np.lib.format.open_memmap(job['out_path'], mode='r+')

    ci, chunk, halo, ns = job['ci'], job['chunk'], job['halo'], job['ns']
    i0_w = ci * chunk
    i1_w = min(i0_w + chunk, ns)
    i0_r = max(0, i0_w - halo)
    i1_r = min(ns, i1_w + halo)
    left_halo = i0_w - i0_r

    snippet = np.asarray(data[i0_r:i1_r, :], dtype=np.float32).T   # (nc, processed)
    denoised = _cadzow_proc.cadzow_denoiser(
        snippet, h=job['h'], fs=job['fs'],
        rank=job['rank'], niter=job['niter'], fmax=job['fmax'],
        nswx=job['nswx'], gap_threshold=job['gap_threshold'],
        ppca_k=job['ppca_k'], n_jobs=1,
    )
    out[i0_w:i1_w, :] = denoised[:, left_halo:left_halo + (i1_w - i0_w)].T
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
    out_mm = np.lib.format.open_memmap(str(out_npy), mode='w+', dtype=np.float32, shape=(ns, nc))
    del out_mm  # flush header + allocation; workers re-open independently

    # Workers need a file path, not an array.  Memmaps expose .filename; otherwise
    # save a temp file so every subprocess can re-open the data without IPC copies.
    _tmp_input = None
    if hasattr(data, 'filename'):
        data_path = str(data.filename)
    else:
        _tmp_input = out_npy.with_suffix('.tmp_input.npy')
        np.save(_tmp_input, data)
        data_path = str(_tmp_input)

    n_chunks = int(np.ceil(ns / chunk))
    report_every = max(1, n_chunks // 20)
    shared = dict(
        data_path=data_path, out_path=str(out_npy), ns=ns,
        chunk=chunk, halo=halo, h=h, fs=fs,
        rank=rank, niter=niter, fmax=fmax,
        nswx=nswx, gap_threshold=gap_threshold, ppca_k=ppca_k,
    )
    jobs = [{**shared, 'ci': ci} for ci in range(n_chunks)]

    n_done = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_cadzow_worker, job): job['ci'] for job in jobs}
        for fut in as_completed(futures):
            fut.result()   # re-raise any worker exception immediately
            n_done += 1
            if n_done % report_every == 0:
                print(f'  Cadzow {n_done}/{n_chunks} ({100 * n_done / n_chunks:.0f}%)', flush=True)

    if _tmp_input is not None:
        _tmp_input.unlink()

    return np.load(out_npy, mmap_mode='r')


def compress_to_h5(
    cadzow_npy,
    out_h5,
    sglx_meta=None,
    h=None,
    chunk=_COMPRESS_CHUNK,
    overlap=_COMPRESS_OVERLAP,
    epsilon=150.0,
    alpha=28.0,
    fs=250.0,
):
    """
    Compress a Cadzow-denoised .npy into a single HDF5 archive of LFPCompressed chunks.

    Each written chunk of `chunk` samples is extended by `overlap` samples on each side
    before SVD + wavelet-packet compression.  Only the central `chunk` columns of Vh_hat
    are stored, eliminating wavelet-reconstruction boundary artefacts.  Decompressed
    chunks are concatenated without overlap during reading.

    HDF5 layout
    -----------
    /meta             attrs: nc, ns_total, fs, compress_chunk, compress_overlap,
                             epsilon, alpha, sglx_meta (JSON), geometry_x, geometry_y
    /chunks/<i>/      datasets: U_scaled (nc, r), vh_indices (n_kept,) int32,
                                vh_values (n_kept,) float32
                      attrs: ns_original, ns_extended, left_overlap, vh_shape,
                             epsilon, alpha, cr_svd, cr_wp, cr_total, rmse

    Parameters
    ----------
    cadzow_npy : path-like
        Path to the (ns, nc) float32 Cadzow checkpoint (time-first).
    out_h5 : path-like
        Output HDF5 file (created or overwritten).
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

    data = np.load(cadzow_npy, mmap_mode='r')  # (ns, nc) time-first
    ns, nc = data.shape
    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:nc] for k, v in _h.items()}

    out_h5 = Path(out_h5)
    n_chunks = int(np.ceil(ns / chunk))
    report_every = max(1, n_chunks // 20)
    total_cr = 0.0

    with h5py.File(out_h5, 'w') as f:
        mg = f.create_group('meta')
        mg.attrs['nc'] = nc
        mg.attrs['ns_total'] = ns
        mg.attrs['fs'] = fs
        mg.attrs['compress_chunk'] = chunk
        mg.attrs['compress_overlap'] = overlap
        mg.attrs['epsilon'] = epsilon
        mg.attrs['alpha'] = alpha
        mg.attrs['sglx_meta'] = _json.dumps(sglx_meta or {})
        mg.attrs['geometry_x'] = h['x'].astype(np.float32)
        mg.attrs['geometry_y'] = h['y'].astype(np.float32)

        cg = f.create_group('chunks')
        for ci in range(n_chunks):
            i0_w = ci * chunk
            i1_w = min(i0_w + chunk, ns)
            n_w = i1_w - i0_w
            i0_r = max(0, i0_w - overlap)
            i1_r = min(ns, i1_w + overlap)
            left_ov = i0_w - i0_r

            # Transpose to (nc, n_samples) for compress(), which expects channels-first
            snippet = np.asarray(data[i0_r:i1_r, :], dtype=np.float32).T
            ns_ext = snippet.shape[1]
            c = compress(snippet, epsilon=epsilon, alpha=alpha)

            # RMSE on the central n_w samples (guard bands excluded)
            reconstructed = decompress(c)  # (nc, ns_ext)
            rmse = float(np.sqrt(np.mean((snippet[:, left_ov:left_ov + n_w].astype(np.float64)
                                          - reconstructed[:, left_ov:left_ov + n_w].astype(np.float64)) ** 2)))

            # Sparse Vh_hat: two contiguous 1-D arrays (indices, values)
            flat = c.Vh_hat.ravel()
            vh_idx = np.flatnonzero(flat).astype(np.int32)
            vh_vals = flat[vh_idx]

            grp = cg.create_group(str(ci))
            grp.create_dataset('U_scaled',   data=c.U_scaled, compression='gzip', shuffle=True)
            grp.create_dataset('vh_indices', data=vh_idx,  compression='gzip', shuffle=True)
            grp.create_dataset('vh_values',  data=vh_vals, compression='gzip', shuffle=True)
            grp.attrs['vh_shape'] = c.Vh_hat.shape
            grp.attrs['ns_original'] = n_w
            grp.attrs['ns_extended'] = ns_ext
            grp.attrs['left_overlap'] = left_ov
            grp.attrs['epsilon'] = epsilon
            grp.attrs['alpha'] = alpha
            grp.attrs['cr_svd'] = c.cr_svd
            grp.attrs['cr_wp'] = c.cr_wp
            grp.attrs['cr_total'] = c.cr_total
            grp.attrs['rmse'] = rmse
            total_cr += c.cr_total

            if ci % report_every == 0:
                print(f'  Compress {ci + 1}/{n_chunks}  CR={c.cr_total:.0f}  RMSE={rmse * 1e6:.2f} µV')

    print(f'Saved {out_h5}  mean CR={total_cr / n_chunks:.0f}')
    return out_h5


class LFPackReader(_spikeglx.Reader):
    """
    Drop-in spikeglx.Reader for HDF5-packed LFP-compressed files.

    Chunks are decompressed on demand.  No sync trace is available; read() with
    sync=True returns None as the second element.  Data is returned in volts (float32)
    in the same (n_samples, n_channels) convention as spikeglx.Reader.

    Parameters
    ----------
    h5_file : path-like
        HDF5 archive produced by compress_to_h5.

    Examples
    --------
    >>> sr = LFPackReader('lf_compressed.h5')
    >>> data, _ = sr.read_samples(0, 2500)    # (2500, nc) float32, volts
    >>> snippet = sr[0:2500]                  # same, sync omitted
    """

    def __init__(self, h5_file):
        import h5py

        self._h5_file = Path(h5_file)
        self._h5 = None
        self._raw = None   # is_open sentinel (None → closed)
        self.geometry = None
        self.ignore_warnings = False
        self.file_bin = self._h5_file
        self.file_meta_data = None
        self.meta = None   # None → base-class properties fall back to _nc/_fs/_ns
        self.dtype = np.dtype('float32')
        self.ch_file = None

        with h5py.File(self._h5_file, 'r') as f:
            attrs = f['meta'].attrs
            self._nc = int(attrs['nc'])
            self._ns = int(attrs['ns_total'])
            self._fs = float(attrs['fs'])
            self._compress_chunk = int(attrs['compress_chunk'])
            self._n_chunks = len(f['chunks'])
            self.sglx_meta = _json.loads(attrs['sglx_meta'])
            self.geometry = {
                'x': attrs['geometry_x'][:].astype(np.float32),
                'y': attrs['geometry_y'][:].astype(np.float32),
            }

        self._nsync = 0
        # Data is already in volts; s2v = 1.0 for all channels.
        self.channel_conversion_sample2v = {'samples': np.ones(self._nc, dtype=np.float32)}
        self.open()

    def open(self):
        import h5py
        self._h5 = h5py.File(self._h5_file, 'r')
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
    def nc(self):
        return self._nc

    @property
    def ns(self):
        return self._ns

    def read(self, nsel=slice(0, 10000), csel=slice(None), sync=True):
        """
        Decompress and return a sample range.

        Parameters
        ----------
        nsel : slice or int
            Sample selection (Python slice convention).
        csel : slice or array-like
            Channel selection.
        sync : bool
            If True returns (data, None); no sync trace in compressed files.

        Returns
        -------
        data : ndarray (n_samples, n_channels), float32, volts
        sync : None  (only when sync=True)
        """
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
            grp = self._h5[f'chunks/{ci}']
            ns_orig = int(grp.attrs['ns_original'])
            # Reconstruct dense Vh_hat from sparse storage
            vh_shape = tuple(int(x) for x in grp.attrs['vh_shape'])
            Vh_hat = np.zeros(vh_shape, dtype=np.float32)
            Vh_hat.ravel()[grp['vh_indices'][:]] = grp['vh_values'][:]
            c = LFPCompressed(
                U_scaled=grp['U_scaled'][:],
                Vh_hat=Vh_hat,
                ns_original=ns_orig,
                epsilon=float(grp.attrs['epsilon']),
                alpha=float(grp.attrs['alpha']),
                cr_svd=float(grp.attrs['cr_svd']),
                cr_wp=float(grp.attrs['cr_wp']),
                cr_total=float(grp.attrs['cr_total']),
                left_overlap=int(grp.attrs.get('left_overlap', 0)),
                ns_extended=int(grp.attrs.get('ns_extended', ns_orig)),
            )
            pieces.append(decompress(c))  # (nc, ns_chunk_i)

        full = np.concatenate(pieces, axis=1)  # (nc, total_samples)
        start = first_sample - first_chunk * chunk
        data = full[:, start:start + (last_sample - first_sample)]  # (nc, n_req)

        # Transpose to spikeglx convention (n_samples, nc)
        data = data.T.astype(np.float32)
        if not (isinstance(csel, slice) and csel == slice(None)):
            data = data[:, csel]

        if sync:
            return data, None
        return data