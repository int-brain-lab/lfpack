"""Provenance of a compressed file: which software and processing settings wrote it."""

import json
import subprocess
from importlib import metadata
from pathlib import Path

PROVENANCE_ATTRS = ("lfpack_version", "lfpack_commit", "ibl_neuropixel_version", "processing")


def _version(dist):
    """Installed version of a distribution, ``"unknown"`` when it is not installed as such."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_commit():
    """
    Commit of the lfpack source checkout, e.g. ``947741d-dirty``; ``""`` for a regular install.

    A wheel carries no git information, and a site-packages directory that happens to sit inside another repository
    (a virtualenv in a monorepo) must not report that repository's commit: the checkout has to hold ``src/lfpack``.
    """
    pkg_dir = Path(__file__).resolve().parent
    try:
        git = ["git", "-C", str(pkg_dir)]
        top = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        if Path(top.stdout.strip()).joinpath("src", "lfpack").resolve() != pkg_dir:
            return ""
        out = subprocess.run([*git, "describe", "--always", "--dirty"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(processing=None):
    """
    Software versions and processing settings to store in a ``meta`` group.

    Parameters
    ----------
    processing : dict or None
        Settings of the pipeline that produced the data (e.g. the Cadzow keyword arguments); stored as JSON.

    Returns
    -------
    dict
        ``meta.attrs`` entries: `PROVENANCE_ATTRS`.  ``lfpack_commit`` is empty unless lfpack runs from a git checkout.
    """
    return {
        "lfpack_version": _version("lfpack"),
        "lfpack_commit": _git_commit(),
        "ibl_neuropixel_version": _version("ibl-neuropixel"),
        "processing": json.dumps(processing or {}, default=str, sort_keys=True),
    }
