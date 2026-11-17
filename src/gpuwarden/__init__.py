"""gpuwarden — lifecycle control for rented RunPod GPU pods."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gpuwarden")  # pyproject.toml is the only place the number lives
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
