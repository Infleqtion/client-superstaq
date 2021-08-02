import os
from typing import Any
from unittest.mock import patch

import cirq
import pytest

import cirq_superstaq as css


def test__get_api_url() -> None:
    assert css.applications._get_api_url() == css.API_URL

    with patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "https://127.0.0.1:5000"}):
        assert css.applications._get_api_url() == "https://127.0.0.1:5000"


def test__get_headers() -> None:
    with pytest.raises(KeyError):
        css.applications._get_headers()

    with patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "foobar"}):
        assert css.applications._get_headers() == {
            "Authorization": "foobar",
            "Content-Type": "application/json",
        }


def test__should_verify_requests() -> None:
    assert css.applications._should_verify_requests() is True

    with patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "https://127.0.0.1:5000"}):
        assert css.applications._should_verify_requests() is False


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "https://127.0.0.1:5000"}, clear=True)
@patch("requests.post")
def test_aqt_compile(mock_post: Any) -> None:
    class MockResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"compiled_circuit": cirq.to_json(cirq.Circuit())}

    mock_post.return_value = MockResponse()
    assert css.aqt_compile(cirq.Circuit()) == cirq.Circuit()
