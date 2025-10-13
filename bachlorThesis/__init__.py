"""Utilities and experiment pipelines for the bachelor thesis project."""

from __future__ import annotations

__all__ = ["__version__"]

try:
    from importlib.metadata import version as _version
except ImportError:  # pragma: no cover - fallback for Python <3.8
    from importlib_metadata import version as _version  # type: ignore

try:
    __version__ = _version("bachlorThesis")
except Exception:
    # Package is likely being used in a local, editable checkout
    __version__ = "0.0.0"
