# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

from general_superstaq import _models


def test_user_token_response() -> None:
    _models.UserTokenResponse(
        email="cameron.booker@infleqtion.com",
        token="abc",
    )