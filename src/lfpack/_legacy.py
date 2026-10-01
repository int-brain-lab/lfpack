"""
Format-1 (one HDF5 group per codec chunk) read support, deprecated.

lfpack >= 1.0 only writes format 2.  Format-1 archives stay readable and ``upgrade_h5``
converts them; everything specific to format 1 lives in this module so that dropping it
is a matter of deleting the file and the ``format_version == 1`` branch of ``LFPackReader``.

Format-1 layout::

    /<recording>/<scale>/chunks/<i>/   datasets: U_scaled (nc, r) float32,
                                                 vh_indices (n_kept,) int32,
                                                 vh_values (n_kept,) float32
                                       attrs: vh_shape, ns_original, ns_extended,
                                              left_overlap, epsilon, alpha, cr_svd,
                                              cr_wp, cr_total, rmse

The legacy flat layout (``meta`` and ``chunks`` at the file root) is format 1 as well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from lfpack import _container
from lfpack._core import _H5_LIBVER, _WP_MAXLEVEL, _WP_WAVELET, _transcopy_group

_ATTRS = ("ns_original", "ns_extended", "left_overlap", "epsilon", "alpha", "cr_svd", "cr_wp", "cr_total", "rmse")


class ChunkGroupReader:
    """
    Random-access reader of format-1 per-chunk groups.

    Parameters
    ----------
    chunks_group : h5py.Group
        The ``chunks`` group of an open file.
    """

    def __init__(self, chunks_group):
        self._cg = chunks_group

    def __len__(self):
        return len(self._cg)

    def results(self, ci):
        """Chunk ``ci`` as the per-chunk dict accepted by the writers."""
        grp = self._cg[str(ci)]
        res = {name: grp[name][()] for name in ("U_scaled", "vh_indices", "vh_values")}
        res["vh_shape"] = tuple(int(x) for x in grp.attrs["vh_shape"])
        ns_original = int(grp.attrs["ns_original"])
        defaults = {"left_overlap": 0, "ns_extended": ns_original, "rmse": np.nan}
        for name in _ATTRS:
            res[name] = grp.attrs.get(name, defaults.get(name))
        return res

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
        res = self.results(ci)
        Vh_hat = np.zeros(res["vh_shape"], dtype=np.float32)
        Vh_hat.ravel()[res["vh_indices"]] = res["vh_values"]
        kw = {name: res[name] for name in ("epsilon", "alpha", "cr_svd", "cr_wp", "cr_total")}
        return dict(
            U_scaled=res["U_scaled"],
            Vh_hat=Vh_hat,
            ns_original=int(res["ns_original"]),
            left_overlap=int(res["left_overlap"]),
            ns_extended=int(res["ns_extended"]),
            **{k: float(v) for k, v in kw.items()},
        )


def upgrade_h5(src_h5, dst_h5, basis_size=_container.BASIS_SIZE):
    """
    Rewrite a format-1 archive in format 2, without re-compressing.

    Every format-1 scale is repacked into the flat ``codec`` group with a shared spatial
    basis; the temporal coefficients are kept as they are (minus those that only reconstruct
    the guard bands), so the only loss is the basis projection (< 0.01 dB SNR at m = 32).
    Everything else (meta, sync datasets, saturation tables, format-2 scales) is copied as is.

    Parameters
    ----------
    src_h5 : path-like
        Source multi-recording archive (``/<recording>/<scale>/...`` layout).
    dst_h5 : path-like
        Output archive (always created fresh).
    basis_size : int
        Number of shared spatial basis vectors.  Default 32.

    Returns
    -------
    Path
        Resolved path to *dst_h5*.
    """
    import h5py

    dst_h5 = Path(dst_h5)
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w", libver=_H5_LIBVER) as dst:
        if "meta" in src:
            raise ValueError("legacy flat layout (meta at the file root) has no recording name to upgrade")
        for recording in tqdm(list(src.keys()), desc=dst_h5.stem, unit="PID"):
            for name, item in src[recording].items():
                if not (isinstance(item, h5py.Group) and "chunks" in item):
                    if isinstance(item, h5py.Group):
                        _transcopy_group(item, dst.require_group(f"{recording}/{name}"))
                    else:
                        _transcopy_group_item(src[recording], dst.require_group(recording), name)
                    continue
                scale = dst.require_group(f"{recording}/{name}")
                _transcopy_group(item["meta"], scale.require_group("meta"))
                reader = ChunkGroupReader(item["chunks"])
                results = [reader.results(ci) for ci in range(len(reader))]
                _container.write_codec(scale, results, _WP_WAVELET, _WP_MAXLEVEL, basis_size=basis_size)
    return dst_h5.resolve()


def _transcopy_group_item(src_group, dst_group, name):
    """Transcode the single dataset ``src_group[name]`` into ``dst_group``."""
    item = src_group[name]
    kw = dict(compression=item.compression, compression_opts=item.compression_opts, shuffle=item.shuffle)
    if item.chunks and all(c <= s for c, s in zip(item.chunks, item.shape)):
        kw["chunks"] = item.chunks
    ds = dst_group.create_dataset(name, data=item[()], **{k: v for k, v in kw.items() if v})
    for key, val in item.attrs.items():
        ds.attrs[key] = val
