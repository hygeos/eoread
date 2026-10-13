from core.tests.conftest import *

import os
from pathlib import Path

import pytest


def require_sample(varname: str) -> Path:
    """
    Return the sample path stored in environment variable `varname`,
    or skip the test if the sample is not available.

    Tests run on real products only; see tests/README.md for the data
    manifest (env vars, expected sample locations). Missing data must
    produce a clean skip, never an error cascade.
    """
    path = os.environ.get(varname, '')
    if not path or not Path(path).exists():
        pytest.skip(f'sample for {varname} is missing (see tests/README.md)')
    return Path(path)
