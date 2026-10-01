"""
Flat, random-access HDF5 container for lfpack codec chunks (format version 2).

Format 1 stored every 8 s codec chunk as its own HDF5 group holding three small
gzip datasets and a dozen attributes, which costs ~9.5 kB of HDF5 object overhead
per chunk (~20 % of a typical file).  Format 2 concatenates the chunks into a few
flat datasets plus a per-chunk table, so the overhead is paid once per scale:

    /<recording>/<scale>/codec/
        chunk_table   (n_chunks,) compound: rank, n_kept, vh_cols, ns_original,
                      ns_extended, left_overlap, cr_svd, cr_wp, cr_total, rmse
        u_values      (sum nc * rank,) float32   U_scaled of each chunk, C-order (nc, rank)
        vh_deltas     (sum n_kept,) uint16|uint32  first differences of the flat
                      Vh_hat indices, restarted at every chunk (so the first value of
                      a chunk is its absolute first index)
        vh_values     (sum n_kept,) float32

Offsets are the cumulative sums of ``nc * rank`` and ``n_kept``, so reading one codec
chunk touches a single HDF5 storage chunk or two per dataset: random access is kept.
The indices shrink because deltas are small (uint16 when they fit) and compress well
after shuffle + gzip.

At write time, Vh_hat coefficients whose synthesis function is zero over the central
(written) samples are dropped: they only reconstruct guard-band samples that the
decoder trims anyway, so the decoded output is unchanged.
"""

from __future__ import annotations

import functools

import numpy as np
import pywt

FORMAT_VERSION = 2
# Elements per HDF5 storage chunk of the flat datasets (64 kB of float32), about one
# codec chunk's worth of coefficients: a random read decompresses at most a few of them.
_STORAGE_CHUNK = 16384
_CHUNK_TABLE_DTYPE = np.dtype(
    [
        ("rank", np.uint16),
        ("n_kept", np.uint32),
        ("vh_cols", np.uint32),
        ("ns_original", np.int32),
        ("ns_extended", np.int32),
        ("left_overlap", np.int32),
        ("cr_svd", np.float32),
        ("cr_wp", np.float32),
        ("cr_total", np.float32),
        ("rmse", np.float32),
    ]
)


@functools.lru_cache(maxsize=32)
def central_slot_mask(ns_extended, left_overlap, ns_original, wavelet, maxlevel, time_domain=False):
    """
    Flag the Vh_hat columns that contribute to the central (written) samples.

    Parameters
    ----------
    ns_extended : int
        Number of samples the chunk was compressed over (written chunk + guard bands).
    left_overlap : int
        Guard-band samples on the left of the written chunk.
    ns_original : int
        Number of written samples.
    wavelet : str
        Wavelet-packet family used by the codec.
    maxlevel : int
        Wavelet-packet decomposition level used by the codec.
    time_domain : bool
        True when Vh_hat holds time-domain rows (codec run with ``alpha == 0``).

    Returns
    -------
    ndarray of bool, (vh_cols,)
        True where the column's synthesis function is non-zero on
        ``[left_overlap, left_overlap + ns_original)``.  Read-only (cached).
    """
    center = slice(left_overlap, left_overlap + ns_original)
    if time_domain:
        mask = np.zeros(ns_extended, dtype=bool)
        mask[center] = True
    else:
        wp = pywt.WaveletPacket(data=np.zeros(ns_extended), wavelet=wavelet, maxlevel=maxlevel)
        nodes = wp.get_level(maxlevel, "natural")
        sizes = [len(node.data) for node in nodes]
        mask = np.zeros(sum(sizes), dtype=bool)
        j = 0
        # synthesis is linear: coefficient j contributes coeff × (reconstruction of a unit
        # impulse in slot j), so an impulse that is exactly zero on the center can be dropped
        for node, size in zip(nodes, sizes):
            for i in range(size):
                for nd, sz in zip(nodes, sizes):
                    nd.data = np.zeros(sz)
                node.data[i] = 1.0
                mask[j] = np.any(wp.reconstruct(update=False)[:ns_extended][center] != 0)
                j += 1
    mask.flags.writeable = False
    return mask


def write_codec(scale_group, results, wavelet, maxlevel):
    """
    Write compressed chunks to ``<scale_group>/codec`` in format 2.

    Parameters
    ----------
    scale_group : h5py.Group
        The ``/<recording>/<scale>`` group; its ``meta`` child gets ``format_version``.
    results : sequence of dict
        One dict per codec chunk, in time order, with keys ``U_scaled`` (nc, r),
        ``vh_indices`` (flat indices into Vh_hat), ``vh_values``, ``vh_shape``,
        ``ns_original``, ``ns_extended``, ``left_overlap``, ``alpha``, ``cr_svd``,
        ``cr_wp``, ``cr_total``, ``rmse``.
    wavelet : str
        Wavelet-packet family used by the codec.
    maxlevel : int
        Wavelet-packet decomposition level used by the codec.
    """
    table = np.zeros(len(results), dtype=_CHUNK_TABLE_DTYPE)
    u_parts, delta_parts, value_parts = [], [], []
    for ci, res in enumerate(results):
        rank, vh_cols = (int(x) for x in res["vh_shape"])
        mask = central_slot_mask(
            int(res["ns_extended"]),
            int(res["left_overlap"]),
            int(res["ns_original"]),
            wavelet,
            maxlevel,
            time_domain=float(res["alpha"]) == 0.0,
        )
        idx = np.asarray(res["vh_indices"], dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        idx, values = idx[order], np.asarray(res["vh_values"], dtype=np.float32)[order]
        keep = mask[idx % vh_cols]
        idx, values = idx[keep], values[keep]
        u_parts.append(np.asarray(res["U_scaled"], dtype=np.float32).ravel())
        delta_parts.append(np.diff(idx, prepend=0))
        value_parts.append(values)
        table[ci] = (
            rank,
            idx.size,
            vh_cols,
            res["ns_original"],
            res["ns_extended"],
            res["left_overlap"],
            res["cr_svd"],
            res["cr_wp"],
            res["cr_total"],
            res["rmse"],
        )

    deltas = np.concatenate(delta_parts) if delta_parts else np.zeros(0, np.int64)
    delta_dtype = np.uint16 if deltas.size == 0 or deltas.max() <= np.iinfo(np.uint16).max else np.uint32
    cg = scale_group.create_group("codec")
    cg.create_dataset("chunk_table", data=table)
    for name, parts, dtype in (
        ("u_values", u_parts, np.float32),
        ("vh_deltas", [deltas], delta_dtype),
        ("vh_values", value_parts, np.float32),
    ):
        data = np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)
        kw = dict(chunks=(min(_STORAGE_CHUNK, data.size),), compression="gzip", shuffle=True) if data.size else {}
        cg.create_dataset(name, data=data, **kw)
    scale_group["meta"].attrs["format_version"] = FORMAT_VERSION


class CodecReader:
    """
    Random-access reader of the format-2 ``codec`` group of one scale.

    Parameters
    ----------
    scale_group : h5py.Group
        The ``/<recording>/<scale>`` group of an open file.
    """

    def __init__(self, scale_group):
        self._cg = scale_group["codec"]
        self._alpha = float(scale_group["meta"].attrs["alpha"])
        self._epsilon = float(scale_group["meta"].attrs["epsilon"])
        self._nc = int(scale_group["meta"].attrs["nc"])
        self.table = self._cg["chunk_table"][()]
        self._u_offsets = np.concatenate([[0], np.cumsum(self.table["rank"].astype(np.int64) * self._nc)])
        self._vh_offsets = np.concatenate([[0], np.cumsum(self.table["n_kept"].astype(np.int64))])

    def __len__(self):
        return self.table.size

    def chunk(self, ci):
        """
        Load one codec chunk.

        Parameters
        ----------
        ci : int
            Chunk index.

        Returns
        -------
        dict
            Keyword arguments for ``lfpack.LFPCompressed``.
        """
        row = self.table[ci]
        rank = int(row["rank"])
        u = self._cg["u_values"][self._u_offsets[ci] : self._u_offsets[ci + 1]]
        v0, v1 = self._vh_offsets[ci], self._vh_offsets[ci + 1]
        flat = np.cumsum(self._cg["vh_deltas"][v0:v1], dtype=np.int64)
        Vh_hat = np.zeros((rank, int(row["vh_cols"])), dtype=np.float32)
        Vh_hat.ravel()[flat] = self._cg["vh_values"][v0:v1]
        return dict(
            U_scaled=u.reshape(self._nc, rank),
            Vh_hat=Vh_hat,
            ns_original=int(row["ns_original"]),
            epsilon=self._epsilon,
            alpha=self._alpha,
            cr_svd=float(row["cr_svd"]),
            cr_wp=float(row["cr_wp"]),
            cr_total=float(row["cr_total"]),
            left_overlap=int(row["left_overlap"]),
            ns_extended=int(row["ns_extended"]),
        )
