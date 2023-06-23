# Note: this try/except is necessary to support python3.7, which does not have importlib.metadata
try:
    import importlib.metadata as importlib_metadata  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]  # pragma: no cover

__version__ = importlib_metadata.version(__package__)
