"""Time-Aligned Peaks (TAP)

A package for plotting multi-series line charts with a synchronized per-timestamp
peak/overlap timeline panel.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("time-aligned-peaks")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
