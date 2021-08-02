import os
from typing import Any
from unittest.mock import patch

import cirq
import pytest

import cirq_superstaq
from cirq_superstaq import API_URL


def test_get_api_url() -> None:
    assert cirq_superstaq.applications._get_api_url() == API_URL

    with patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "https://127.0.0.1:5000"}):
        assert cirq_superstaq.applications._get_api_url() == "https://127.0.0.1:5000"


def test_get_headers() -> None:
    with pytest.raises(KeyError):
        cirq_superstaq.applications._get_headers()

    with patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "foobar"}):
        assert cirq_superstaq.applications._get_headers() == {
            "Authorization": "foobar",
            "Content-Type": "application/json",
        }


def test_should_verify_requests() -> None:
    assert cirq_superstaq.applications._should_verify_requests() is True

    with patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "https://127.0.0.1:5000"}):
        assert cirq_superstaq.applications._should_verify_requests() is False


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "https://127.0.0.1:5000"}, clear=True)
@patch("requests.post")
def test_aqt_compile(mock_post: Any) -> None:
    class MockResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"compiled_circuit": cirq.to_json(cirq.Circuit())}

    mock_post.return_value = MockResponse()
    assert cirq_superstaq.aqt_compile(cirq.Circuit()) == cirq.Circuit()
