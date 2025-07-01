from __future__ import annotations

import pydantic
import pytest

from general_superstaq import _models

"""This test file will be updated once the new server (v0.3.0) client is implemented.
For now, a single test is sufficient to check that `pydantic` is loading models and working as
expected.
"""


def test_user_token_response() -> None:
    _models.UserTokenResponse(
        email="cameron.booker@infleqtion.com",
        token="abc",
    )
    with pytest.raises(pydantic.ValidationError):
        _models.UserTokenResponse(
            email="this is not an email",
            token="abc",
        )
