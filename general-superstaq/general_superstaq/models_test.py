from __future__ import annotations

import pydantic
import pytest

import general_superstaq as gss

"""This test file will be updated once the new server (v0.3.0) client is implemented.
For now, a single test is sufficient to check that `pydantic` is loading models and working as
expected.
"""


def test_user_token_response() -> None:
    gss.models.UserTokenResponse(
        email="valid.email@infleqtion.com",
        token="abc",
    )
    with pytest.raises(pydantic.ValidationError):
        gss.models.UserTokenResponse(
            email="this is not an email",
            token="abc",
        )


def test_external_provider_credentials() -> None:
    credentials = gss.models.ExternalProviderCredentials()
    assert credentials.cq_token is None
    assert credentials.cq_project_id is None
    assert credentials.cq_org_id is None

    # New-style CQ credentials
    options_new = {"cq_token": "token", "cq_project_id": "123", "cq_org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_new)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"

    # Old-style CQ credentials
    options_old = {"cq_token": {"access_token": "token"}, "project_id": "123", "org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_old)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"


def test_worker_task_results() -> None:
    with pytest.raises(ValueError, match=r"must return both"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.COMPLETED,
            measurements={"000": [0, 1, 3], "101": [2]},
        )

    with pytest.raises(ValueError, match=r"same length"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=4,
            measurements={"000": [0, 1, 3], "1010": [2]},
        )

    with pytest.raises(ValueError, match=r"valid bitstrings"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=4,
            measurements={"000": [0, 1, 3], "xyz": [2]},
        )

    with pytest.raises(ValueError, match=r"Not all successful shots have a measurement"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=4,
            measurements={"000": [0, 3], "101": [2]},
        )

    with pytest.raises(ValueError, match=r"cannot return a status"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.AWAITING_COMPILE,
        )

    with pytest.raises(ValueError, match=r"cannot return results unless"):
        _ = gss.models.WorkerTaskResults(
            circuit_ref="1234",
            status=gss.models.CircuitStatus.FAILED,
            successful_shots=4,
            measurements={"000": [0, 1, 3], "101": [2]},
        )

    _ = gss.models.WorkerTaskResults(
        circuit_ref="1234",
        status=gss.models.CircuitStatus.COMPLETED,
        successful_shots=4,
        measurements={"000": [0, 1, 3], "101": [2]},
    )

    _ = gss.models.WorkerTaskResults(
        circuit_ref="1234",
        status=gss.models.CircuitStatus.FAILED,
        status_message="message",
    )
