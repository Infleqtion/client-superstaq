from __future__ import annotations

import datetime

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


def test_job_data_consistent_number_of_circuits() -> None:
    # First test a correct model validates
    _ = gss.models.JobData(
        job_type=gss.models.JobType.SIMULATE,
        statuses=[
            gss.models.CircuitStatus.RECEIVED,
            gss.models.CircuitStatus.RECEIVED,
        ],
        status_messages=[None, None],
        user_email="a@b.com",
        target="example",
        provider_id=[None, None],
        num_circuits=2,
        compiled_circuits=["c1", "c2"],
        input_circuits=["ic1", "ic2"],
        pulse_gate_circuits=[None, None],
        counts=[None, None],
        results_dicts=[None, None],
        num_qubits=[1, 1],
        shots=[1, 1],
        dry_run=True,
        submission_timestamp=datetime.datetime(2000, 1, 1),
        last_updated_timestamp=[None, None],
        initial_logical_to_physicals=[None, None],
        final_logical_to_physicals=[None, None],
        circuit_type=gss.models.CircuitType.CIRQ,
        physical_qubits=[None, None],
        logical_qubits=[None, None],
        tags=["tag"],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Field compiled_circuits does not contain the correct number of elements. "
            r"Expected 2 but found 3."
        ),
    ):
        _ = gss.models.JobData(
            job_type=gss.models.JobType.SIMULATE,
            statuses=[
                gss.models.CircuitStatus.RECEIVED,
                gss.models.CircuitStatus.RECEIVED,
            ],
            status_messages=[None, None],
            user_email="a@b.com",
            target="example",
            provider_id=[None, None],
            num_circuits=2,
            compiled_circuits=["c1", "c2", "c3"],
            input_circuits=["ic1", "ic2"],
            pulse_gate_circuits=[None, None],
            counts=[None, None],
            results_dicts=[None, None],
            num_qubits=[1, 1],
            shots=[1, 1],
            dry_run=True,
            submission_timestamp=datetime.datetime(2000, 1, 1),
            last_updated_timestamp=[None, None],
            initial_logical_to_physicals=[None, None],
            final_logical_to_physicals=[None, None],
            circuit_type=gss.models.CircuitType.CIRQ,
            physical_qubits=[None, None],
            logical_qubits=[None, None],
        )
