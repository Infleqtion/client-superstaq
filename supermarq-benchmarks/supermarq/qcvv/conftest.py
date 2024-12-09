from __future__ import annotations

import os
from collections.abc import Generator
from unittest import mock

import pytest


@pytest.fixture(scope="session", autouse=True)
def _patch_tqdm() -> Generator[None, None, None]:
    """Disable progress bars during tests."""
    with mock.patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        yield
