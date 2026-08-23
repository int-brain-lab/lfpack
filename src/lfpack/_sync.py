"""Non-linear ALF sync knot storage for lfpack ``meta`` HDF5 groups.

Each scale's ``meta`` group carries two related but independent tiers of sync
metadata:

- ``sync_samples`` / ``sync_times`` (float64 arrays) — the raw ALF sync knot
  pairs (sample index <-> time), stored verbatim, in that scale's own native
  sample-index units (rate-scaled per scale, mirroring the existing
  ``fs_sync`` convention).
- ``t0_sync`` / ``fs_sync`` (float64 scalars) — a least-squares affine summary
  derived from the knots.  Downstream consumers (``LFPackReader.t0``/``.fs``,
  ``qc_report.py`` reading ``meta.attrs`` directly via h5py, the LFP encoder
  pipeline) require these two to always remain plain scalars, so they are
  written here alongside the knots rather than only living in the knot pair.

This module centralises the attr names and the write/clear/validate logic so
the reader (``lfpack._core.LFPackReader``) and the metadata-attachment
pipeline share exactly one definition.
"""

from __future__ import annotations

import numpy as np

#: HDF5 attr names written/cleared as a unit by `write_sync_attrs`/`clear_sync_attrs`.
SYNC_ATTRS = ("sync_samples", "sync_times", "t0_sync", "fs_sync")


def _validate_knots(sample_knots, time_knots) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a pair of ALF sync knot arrays.

    Returns
    -------
    sample_knots, time_knots : ndarray, float64

    Raises
    ------
    ValueError
        If the arrays are not equal-length 1-D arrays of at least 2 knots, are
        not all-finite, or are not strictly increasing.  Strict (not weak)
        monotonicity is required because `scipy.interpolate.interp1d` silently
        returns NaN for repeated x-values rather than raising.
    """
    sample_knots = np.asarray(sample_knots, dtype=np.float64)
    time_knots = np.asarray(time_knots, dtype=np.float64)
    if sample_knots.ndim != 1 or sample_knots.shape != time_knots.shape:
        raise ValueError(
            f"sample_knots and time_knots must be 1-D arrays of equal length, "
            f"got shapes {sample_knots.shape} and {time_knots.shape}"
        )
    if sample_knots.size < 2:
        raise ValueError(f"need at least 2 sync knots, got {sample_knots.size}")
    if not (np.all(np.isfinite(sample_knots)) and np.all(np.isfinite(time_knots))):
        raise ValueError("sync knots must be all-finite")
    if not np.all(np.diff(sample_knots) > 0):
        raise ValueError("sample_knots must be strictly increasing")
    if not np.all(np.diff(time_knots) > 0):
        raise ValueError("time_knots must be strictly increasing")
    return sample_knots, time_knots


def write_sync_attrs(meta_group, sample_knots, time_knots) -> tuple[float, float]:
    """Write ALF sync knots plus a derived affine summary to a ``meta`` HDF5 group.

    Parameters
    ----------
    meta_group : h5py.Group
        The scale's ``meta`` group (``f'{recording}/{scale:02d}/meta'``).
    sample_knots : array_like
        Sample-index knots, in this scale's own native sample-index units
        (rate-scaled per scale, matching the existing ``fs_sync`` convention).
    time_knots : array_like
        Session-clock time knots [s], one per ``sample_knots`` entry.

    Returns
    -------
    t0_sync, fs_sync : float
        The least-squares affine derived from the knots and written alongside
        them, so ``.t0``/``.fs`` and any direct-attrs reader (e.g.
        ``qc_report.py``) keep seeing a plain scalar summary.

    Raises
    ------
    ValueError
        See `_validate_knots`.
    """
    sample_knots, time_knots = _validate_knots(sample_knots, time_knots)
    slope, intercept = np.polyfit(sample_knots, time_knots, 1)
    t0_sync = float(intercept)
    fs_sync = float(1.0 / slope)
    meta_group.attrs["sync_samples"] = sample_knots
    meta_group.attrs["sync_times"] = time_knots
    meta_group.attrs["t0_sync"] = t0_sync
    meta_group.attrs["fs_sync"] = fs_sync
    return t0_sync, fs_sync


def clear_sync_attrs(meta_group) -> None:
    """Delete all four sync attrs (`SYNC_ATTRS`) from a ``meta`` HDF5 group.

    No-op for attrs that are already absent.
    """
    for key in SYNC_ATTRS:
        if key in meta_group.attrs:
            del meta_group.attrs[key]
