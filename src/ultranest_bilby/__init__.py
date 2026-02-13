"""Interface and plugin for using ultranest in bilby."""

from importlib.metadata import PackageNotFoundError, version


from .plugin import Ultranest

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"


__all__ = ["Ultranest"]
