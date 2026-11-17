"""pyproject.toml is the only place the version number is written."""
import re
from pathlib import Path

import gpuwarden

ROOT = Path(__file__).resolve().parent.parent


def test_version_comes_from_pyproject():
    # regex, not tomllib: the project still supports Python 3.10
    declared = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert gpuwarden.__version__ == declared


def test_no_second_hardcoded_copy():
    source = (ROOT / "src/gpuwarden/__init__.py").read_text()
    assert not re.search(r"""__version__\s*=\s*["']\d+\.\d""", source)
