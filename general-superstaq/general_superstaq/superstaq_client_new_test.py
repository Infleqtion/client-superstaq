# Copyright 2021 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
from unittest import mock
import uuid

import pytest
import requests

import general_superstaq as gss
from general_superstaq.testing import RETURNED_TARGETS, TARGET_LIST
from general_superstaq.superstaq_client import _SuperstaqClient_v0_3_0

API_VERSION = "v0.3.0"
EXPECTED_HEADERS = {
    "Authorization": "to_my_heart",
    "Content-Type": "application/json",
    "X-Client-Version": API_VERSION,
    "X-Client-Name": "general-superstaq",
}


@pytest.fixture
def default_client() -> _SuperstaqClient_v0_3_0:
    return gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )


def test_get_client() -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    assert isinstance(client, gss.superstaq_client._SuperstaqClient)

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    assert isinstance(client, gss.superstaq_client._SuperstaqClient_v0_3_0)

    with pytest.raises(RuntimeError):
        gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="to_my_heart",
            api_version="bad_version",
        )


def test_superstaq_client_str_and_repr(default_client: _SuperstaqClient_v0_3_0) -> None:
    assert str(default_client) == (
        "Client version v0.3.0 with host=http://example.com/v0.3.0 and " "name=general-superstaq"
    )
    assert str(eval(repr(default_client))) == str(default_client)


def test_superstaq_client_args() -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq_token",
        ibmq_channel="ibm_quantum",
        ibmq_instance="instance",
        ibmq_token="ibmq_token",
    )
    assert client.headers == {
        "Authorization": "to_my_heart",
        "Content-Type": "application/json",
        "X-Client-Name": "general-superstaq",
        "X-Client-Version": "v0.3.0",
        "cq_token": "cq_token",
        "ibmq_channel": "ibm_quantum",
        "ibmq_instance": "instance",
        "ibmq_token": "ibmq_token",
    }

    with pytest.raises(ValueError, match="must be either 'ibm_cloud' or 'ibm_quantum'"):
        _ = gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_version="v0.3.0",
            api_key="to_my_heart",
            ibmq_channel="foo",
        )


def test_general_superstaq_exception_str() -> None:
    ex = gss.SuperstaqServerException("err.", status_code=501)
    assert str(ex) == "err. (Status code: 501)"


def test_warning_from_server(default_client: _SuperstaqClient_v0_3_0) -> None:
    warning = {"message": "WARNING!", "category": "SuperstaqWarning"}

    with mock.patch("requests.Session.get", ok=True) as mock_request:
        mock_request.return_value.json = lambda: {"abc": 123, "warnings": [warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert default_client.get_request("/endpoint") == {"abc": 123}

    with mock.patch("requests.Session.post", ok=True) as mock_request:
        mock_request.return_value.json = lambda: {"abc": 123, "warnings": [warning, warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert default_client.post_request("/endpoint", {}) == {"abc": 123}


@pytest.mark.parametrize("invalid_url", ("url", "http://", "ftp://", "http://"))
def test_superstaq_client_invalid_remote_host(invalid_url: str) -> None:
    with pytest.raises(AssertionError, match="not a valid url"):
        _ = gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host=invalid_url,
            api_key="a",
            api_version="v0.3.0",
        )
    with pytest.raises(AssertionError, match=invalid_url):
        _ = gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host=invalid_url,
            api_key="a",
            api_version="v0.3.0",
        )


def test_superstaq_client_invalid_api_version() -> None:
    with pytest.raises(
        NotImplementedError, match="The version v0.0 it not a supported API version."
    ):
        _ = gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version="v0.0",
        )


def test_superstaq_client_time_travel() -> None:
    with pytest.raises(AssertionError, match="time machine"):
        _ = gss.superstaq_client.get_client(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_version="v0.3.0",
            api_key="a",
            max_retry_seconds=-1,
        )


def test_superstaq_client_attributes() -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        max_retry_seconds=10,
        verbose=True,
    )
    assert client.url == f"http://example.com/{API_VERSION}"
    assert client.headers == EXPECTED_HEADERS
    assert client.max_retry_seconds == 10
    assert client.verbose


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient_v0_3_0._accept_terms_of_use")
@mock.patch("requests.Session.get")
def test_superstaq_client_needs_accept_terms_of_use(
    mock_get: mock.MagicMock,
    mock_accept_terms_of_use: mock.MagicMock,
    default_client: _SuperstaqClient_v0_3_0,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_get_response = mock.MagicMock()
    fake_get_response.ok = False
    fake_get_response.status_code = requests.codes.unauthorized
    fake_get_response.json.return_value = (
        "You must accept the Terms of Use (superstaq.infleqtion.com/terms_of_use)."
    )
    mock_get.return_value = fake_get_response

    mock_accept_terms_of_use.return_value = "YES response required to proceed"

    with mock.patch("builtins.input"):
        with pytest.raises(
            gss.SuperstaqServerException, match="You'll need to accept the Terms of Use"
        ):
            default_client.get_balance()
        assert capsys.readouterr().out == "YES response required to proceed\n"

    fake_authorized_get_response = mock.MagicMock(ok=True)
    fake_authorized_get_response.json.return_value = gss._models.BalanceResponse(
        email="example@infleqtion.com", balance=1234
    ).model_dump(mode="json")

    mock_get.side_effect = [fake_get_response, fake_authorized_get_response]
    mock_accept_terms_of_use.return_value = "Accepted. You can now continue using Superstaq."
    with mock.patch("builtins.input"):
        default_client.get_balance()
        assert capsys.readouterr().out == "Accepted. You can now continue using Superstaq.\n"


@mock.patch("requests.Session.get")
def test_superstaq_client_validate_email_error(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.unauthorized
    mock_get.return_value.json.return_value = "You must validate your registered email."
    with pytest.raises(
        gss.SuperstaqServerException, match="You must validate your registered email."
    ):
        _ = default_client.get_balance()


@mock.patch("requests.Session.post")
def test_supertstaq_client_create_job(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_post.return_value.status_code = requests.codes.ok
    job_id = uuid.UUID("f7eed062-65f4-4c58-930d-19226b44e9a9")
    mock_post.return_value.json.return_value = gss._models.NewJobResponse(
        job_id=job_id, num_circuits=1
    ).model_dump(mode="json")

    response = default_client.create_job(
        serialized_circuits={"cirq_circuits": "World"},
        repetitions=200,
        target="ss_example_qpu",
        method="dry-run",
        cq_token={"@type": "RefreshFlowState", "access_token": "123"},
    )
    assert response == {
        "job_id": uuid.UUID("f7eed062-65f4-4c58-930d-19226b44e9a9"),
        "num_circuits": 1,
    }

    expected_json = gss._models.NewJob(
        job_type=gss._models.JobType.DEVICE_SUBMISSION,
        circuit_type=gss._models.CircuitType.CIRQ,
        target="ss_example_qpu",
        circuits="World",
        shots=200,
        dry_run=True,
    ).model_dump()

    expected_headers = EXPECTED_HEADERS
    expected_headers["cq_token"] = {"@type": "RefreshFlowState", "access_token": "123"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/client/job",
        json=expected_json,
        headers=expected_headers,
        verify=False,
    )


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_unauthorized(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = default_client.create_job(
            serialized_circuits={"cirq_circuits": "World"},
            target="ss_example_qpu",
        )


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_not_found(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_found

    with pytest.raises(gss.SuperstaqServerException):
        _ = default_client.create_job(
            serialized_circuits={"cirq_circuits": "World"}, target="ss_example_qpu"
        )


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_not_retriable(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 501"):
        _ = default_client.create_job(
            serialized_circuits={"cirq_circuits": "World"}, target="ss_example_qpu"
        )


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock(ok=False, status_code=requests.codes.service_unavailable)
    response2 = mock.MagicMock(ok=True)
    response2.json.return_value = gss._models.NewJobResponse(
        job_id=uuid.uuid4(), num_circuits=1
    ).model_dump(mode="json")
    mock_post.side_effect = [response1, response2]
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        verbose=True,
    )
    test_stdout = io.StringIO()
    with contextlib.redirect_stdout(test_stdout):
        _ = client.create_job(
            serialized_circuits={"cirq_circuits": "World"}, target="ss_example_qpu"
        )
    assert test_stdout.getvalue().strip() == "Waiting 0.1 seconds before retrying."
    assert mock_post.call_count == 2


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_retry_request_error(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    response2 = mock.MagicMock(ok=True)
    response2.json.return_value = gss._models.NewJobResponse(
        job_id=uuid.uuid4(), num_circuits=1
    ).model_dump(mode="json")

    mock_post.side_effect = [requests.exceptions.ConnectionError(), response2]
    _ = default_client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")
    assert mock_post.call_count == 2


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_invalid_json(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    response = requests.Response()
    response.status_code = requests.codes.not_implemented
    response._content = b"invalid/json"
    mock_post.return_value = response

    with mock.patch("requests.Session.post", return_value=response):
        with pytest.raises(gss.SuperstaqServerException, match="invalid/json"):
            _ = default_client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_dont_retry_on_timeout(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    response = requests.Response()
    response.status_code = requests.codes.gateway_timeout
    response._content = b"invalid/json"
    mock_post.return_value = response

    with mock.patch("requests.Session.post", return_value=response):
        with pytest.raises(gss.SuperstaqServerException, match="timed out"):
            _ = default_client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_timeout(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.service_unavailable

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        max_retry_seconds=0.2,
    )
    with pytest.raises(TimeoutError):
        _ = client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_json(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.bad_request
    mock_post.return_value.json.return_value = {"message": "foo bar"}

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = default_client.create_job(
            serialized_circuits={"cirq_circuits": "World"},
            repetitions=200,
            target="ss_example_qpu",
        )


@mock.patch("requests.Session.get")
def test_superstaq_client_fetch_jobs(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    job_id = uuid.uuid4()
    job_data = gss._models.JobData(
        job_type=gss._models.JobType.DEVICE_SUBMISSION,
        statuses=[gss._models.CircuitStatus.RUNNING],
        status_messages=[None],
        user_email="example@infleqtion.com",
        target="ss_example_qpu",
        provider_id=["example_id"],
        num_circuits=1,
        compiled_circuit_type=gss._models.CircuitType.CIRQ,
        compiled_circuits=["compiled_circuit"],
        input_circuits=["input_circuits"],
        input_circuit_type=gss._models.CircuitType.QISKIT,
        pulse_gate_circuits=[None],
        counts=[None],
        state_vectors=[None],
        results_dicts=[None],
        num_qubits=[3],
        shots=[100],
        dry_run=False,
        submission_timestamp=datetime.datetime(2000, 1, 1, 0, 0, 0),
        last_updated_timestamp=[datetime.datetime(2000, 1, 1, 0, 1, 0)],
        initial_logical_to_physicals=[None],
        final_logical_to_physicals=[None],
    )
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {job_id: job_data.model_dump(mode="json")}
    response = default_client.fetch_jobs(job_ids=[str(job_id)], cq_token={"access_token": "token"})
    assert response == {job_id: job_data.model_dump()}

    expected_headers = EXPECTED_HEADERS
    expected_headers["cq_token"] = {"access_token": "token"}
    mock_get.assert_called_with(
        f"http://example.com/v0.3.0/client/job?job_id={job_id}",
        headers=expected_headers,
        verify=False,
    )


@mock.patch("requests.Session.get")
def test_superstaq_client_fetch_single_jobs(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    job_id = uuid.uuid4()
    job_data = gss._models.JobData(
        job_type=gss._models.JobType.DEVICE_SUBMISSION,
        statuses=[gss._models.CircuitStatus.RUNNING],
        status_messages=[None],
        user_email="example@infleqtion.com",
        target="ss_example_qpu",
        provider_id=["example_id"],
        num_circuits=1,
        compiled_circuit_type=gss._models.CircuitType.CIRQ,
        compiled_circuits=["compiled_circuit"],
        input_circuits=["input_circuits"],
        input_circuit_type=gss._models.CircuitType.QISKIT,
        pulse_gate_circuits=[None],
        counts=[None],
        state_vectors=[None],
        results_dicts=[None],
        num_qubits=[3],
        shots=[100],
        dry_run=False,
        submission_timestamp=datetime.datetime(2000, 1, 1, 0, 0, 0),
        last_updated_timestamp=[datetime.datetime(2000, 1, 1, 0, 1, 0)],
        initial_logical_to_physicals=[None],
        final_logical_to_physicals=[None],
    )
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = job_data.model_dump(mode="json")
    response = default_client.fetch_single_job(str(job_id), cq_token={"access_token": "token"})
    assert response == job_data

    expected_headers = EXPECTED_HEADERS
    expected_headers["cq_token"] = {"access_token": "token"}
    mock_get.assert_called_with(
        f"http://example.com/v0.3.0/client/job/{job_id}",
        headers=expected_headers,
        verify=False,
    )


@mock.patch("requests.Session.get")
def test_superstaq_client_get_balance(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {
        "email": "example@infleqtion.com",
        "balance": 123.4567,
    }
    response = default_client.get_balance()
    assert response == {"balance": 123.4567}

    mock_get.assert_called_with(
        "http://example.com/v0.3.0/client/balance",
        headers=EXPECTED_HEADERS,
        verify=False,
    )


@mock.patch("requests.Session.get")
def test_superstaq_client_get_version(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.headers = {"superstaq_version": "1.2.3"}
    response = default_client.get_superstaq_version()
    assert response == {"superstaq_version": "1.2.3"}

    mock_get.assert_called_with(f"http://example.com/{API_VERSION}")


@mock.patch("requests.Session.post")
def test_add_new_user(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:
    default_client.add_new_user({"name": "Marie Curie", "email": "mc@gmail.com"})
    mock_post.assert_called_with(
        "http://example.com/v0.3.0/client/user",
        headers=EXPECTED_HEADERS,
        json={"name": "Marie Curie", "email": "mc@gmail.com"},
        verify=False,
    )


@mock.patch("requests.Session.put")
def test_update_user_balance(
    mock_put: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:

    default_client.update_user_balance({"email": "mc@gmail.com", "balance": 5.00})

    expected_json = {"balance": 5.00}
    mock_put.assert_called_with(
        "http://example.com/v0.3.0/client/user/mc@gmail.com",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.Session.put")
def test_update_user_role(
    mock_put: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    default_client.update_user_role({"email": "mc@gmail.com", "role": "free_trial"})

    expected_json = {"role": "free_trial"}
    mock_put.assert_called_with(
        "http://example.com/v0.3.0/client/user/mc@gmail.com",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.Session.post")
def test_superstaq_client_resource_estimate(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:

    with pytest.raises(NotImplementedError):
        default_client.resource_estimate({"Hello": "1", "World": "2"})


@mock.patch("requests.Session.get")
def test_superstaq_client_get_targets(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = [
        gss._models.TargetModel(
            target_name="example",
            supports_submit=True,
            supports_submit_qubo=True,
            supports_compile=True,
            available=True,
            retired=False,
            simulator=False,
        ).model_dump(mode="json")
    ]

    response = default_client.get_targets(ibmq_token="ibmq_token")
    assert response == [
        gss.typing.Target(
            target="example",
            supports_submit=True,
            supports_submit_qubo=True,
            supports_compile=True,
            available=True,
            retired=False,
            simulator=False,
        )
    ]
    mock_get.assert_called_once_with(
        "http://example.com/v0.3.0/client/targets",
        headers={"ibmq_token": "ibmq_token", **EXPECTED_HEADERS},
        verify=False,
    )

    response = default_client.get_targets(simulator=True)
    mock_get.assert_called_with(
        "http://example.com/v0.3.0/client/targets?simulator=True",
        headers=EXPECTED_HEADERS,
        verify=False,
    )

    response = default_client.get_targets(supports_submit=True, supports_compile=True)
    mock_get.assert_called_with(
        "http://example.com/v0.3.0/client/targets?supports_submit=True&supports_compile=True",
        headers=EXPECTED_HEADERS,
        verify=False,
    )


@mock.patch("requests.Session.post")
def test_superstaq_client_get_my_targets(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    with pytest.raises(NotImplementedError):
        default_client.get_my_targets()


@mock.patch("requests.Session.get")
def test_superstaq_client_fetch_jobs_unauthorized(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.unauthorized

    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = default_client.fetch_jobs(["job_id"])


@mock.patch("requests.Session.get")
def test_superstaq_client_fetch_jobs_not_found(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.not_found

    with pytest.raises(gss.SuperstaqServerException):
        _ = default_client.fetch_jobs(["job_id"])


@mock.patch("requests.Session.get")
def test_superstaq_client_fetch_jobs_not_retriable(
    mock_get: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.bad_request

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = default_client.fetch_jobs(["job_id"])


@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    _ = client.fetch_jobs(["job_id"])
    assert mock_post.call_count == 2


@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_unauthorized(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.cancel_jobs(["job_id"])


@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_not_found(mock_post: mock.MagicMock) -> None:
    (mock_post.return_value).ok = False
    (mock_post.return_value).status_code = requests.codes.not_found

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException):
        _ = client.cancel_jobs(["job_id"])


@mock.patch("requests.Session.post")
def test_superstaq_client_get_cancel_jobs_retriable(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.bad_request

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.cancel_jobs(["job_id"], cq_token=1)


@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock(ok=False, status_code=requests.codes.service_unavailable)
    response2 = mock.MagicMock(ok=True)
    mock_post.side_effect = [response1, response2]

    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    _ = client.cancel_jobs(["job_id"])
    assert mock_post.call_count == 2


@mock.patch("requests.Session.post")
def test_superstaq_client_aqt_compile(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    with pytest.raises(NotImplementedError):
        default_client.aqt_compile({"Hello": "1", "World": "2"})


@mock.patch("requests.Session.post")
def test_superstaq_client_qscout_compile(
    mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0
) -> None:
    with pytest.raises(NotImplementedError):
        default_client.qscout_compile({"Hello": "1", "World": "2"})


@mock.patch("requests.Session.post")
def test_superstaq_client_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    client.compile(
        {"Hello": "1", "World": "2"},
    )

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/compile"


@mock.patch("requests.Session.post")
def test_superstaq_client_submit_qubo(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:
    example_qubo = {
        ("a",): 2.0,
        ("a", "b"): 1.0,
        ("b", 0): -5,
        (): -3.0,
    }
    target = "ss_unconstrained_simulator"
    repetitions = 10
    with pytest.raises(NotImplementedError):
        default_client.submit_qubo(example_qubo, target, repetitions=repetitions, max_solutions=1)


@mock.patch("requests.Session.post")
def test_superstaq_client_supercheq(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:
    with pytest.raises(NotImplementedError):
        default_client.supercheq([[0]], 1, 1, "cirq_circuits")


@mock.patch("requests.Session.post")
def test_superstaq_client_aces(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:

    with pytest.raises(NotImplementedError):
        default_client.submit_aces(
            target="ss_unconstrained_simulator",
            qubits=[0, 1],
            shots=100,
            num_circuits=10,
            mirror_depth=6,
            extra_depth=4,
            method="dry-run",
            weights=[1, 2],
            tag="test-tag",
            lifespan=10,
            noise={"type": "symmetric_depolarize", "params": (0.01,)},
        )

    with pytest.raises(NotImplementedError):
        default_client.process_aces("id")


@mock.patch("requests.Session.post")
def test_superstaq_client_cb(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:
    with pytest.raises(NotImplementedError):
        default_client.submit_cb(
            target="ss_unconstrained_simulator",
            shots=100,
            serialized_circuits={"circuits": "test_circuit_data"},
            n_channels=6,
            n_sequences=30,
            depths=[2, 4, 6],
            method="dry-run",
            noise={"type": "symmetric_depolarize", "params": (0.01,)},
        )

    with pytest.raises(NotImplementedError):
        default_client.process_cb("id", counts="[{" "}]")


@mock.patch("requests.Session.post")
def test_superstaq_client_dfe(mock_post: mock.MagicMock, default_client: _SuperstaqClient_v0_3_0) -> None:
    with pytest.raises(NotImplementedError):
        default_client.submit_dfe(
            circuit_1={"Hello": "World"},
            target_1="ss_example_qpu",
            circuit_2={"Hello": "World"},
            target_2="ss_example_qpu",
            num_random_bases=5,
            shots=100,
            lifespan=10,
        )

    with pytest.raises(NotImplementedError):
        default_client.process_dfe(["id1", "id2"])



@mock.patch("requests.Session.post")
def test_superstaq_client_aqt_upload_configs(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )

    client.aqt_upload_configs({"pulses": "Hello", "variables": "World"})

    expected_json = {"pulses": "Hello", "variables": "World"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/aqt_configs",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.Session.get")
def test_superstaq_client_aqt_get_configs(mock_get: mock.MagicMock) -> None:
    expected_json = {"pulses": "Hello", "variables": "World"}

    mock_get.return_value.json.return_value = expected_json
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )

    assert client.aqt_get_configs() == expected_json


@mock.patch("requests.Session.post")
def test_superstaq_client_target_info(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
    )
    client.target_info("ss_example_qpu")

    expected_json = {"target": "ss_example_qpu", "options": "{}"}

    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/target_info",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.Session.post")
def test_superstaq_client_target_info_with_credentials(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq-token",
    )
    client.target_info("ss_example_qpu")

    expected_json = {"target": "ss_example_qpu", "options": json.dumps({"cq_token": "cq-token"})}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/target_info",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


def test_find_api_key() -> None:
    # find key in the environment
    with mock.patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "tomyheart"}):
        assert gss.superstaq_client.find_api_key() == "tomyheart"

    # find key in a config file
    with mock.patch.dict(os.environ, SUPERSTAQ_API_KEY=""):
        with mock.patch("pathlib.Path.is_file", return_value=True):
            with mock.patch("builtins.open", mock.mock_open(read_data="tomyheart")):
                assert gss.superstaq_client.find_api_key() == "tomyheart"

    # fail to find an API key :(
    with pytest.raises(EnvironmentError, match="Superstaq API key not specified and not found."):
        with mock.patch.dict(os.environ, SUPERSTAQ_API_KEY=""):
            with mock.patch("pathlib.Path.is_file", return_value=False):
                gss.superstaq_client.find_api_key()


@mock.patch("requests.Session.get")
def test_get_user_info(mock_get: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq-token",
    )
    mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}

    user_info = client.get_user_info()
    mock_get.assert_called_once_with(
        f"http://example.com/{API_VERSION}/user_info",
        headers=EXPECTED_HEADERS,
        verify=False,
    )
    assert user_info == [{"Some": "Data"}]


@mock.patch("requests.Session.get")
def test_get_user_info_query(mock_get: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq-token",
    )
    mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}

    user_info = client.get_user_info(name="Alice")
    mock_get.assert_called_once_with(
        f"http://example.com/{API_VERSION}/user_info?name=Alice",
        headers=EXPECTED_HEADERS,
        verify=False,
    )
    assert user_info == [{"Some": "Data"}]


@mock.patch("requests.Session.get")
def test_get_user_info_query_composite(mock_get: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq-token",
    )
    mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}
    user_info = client.get_user_info(user_id=42, name="Alice")
    mock_get.assert_called_once_with(
        f"http://example.com/{API_VERSION}/user_info?name=Alice&id=42",
        headers=EXPECTED_HEADERS,
        verify=False,
    )
    assert user_info == [{"Some": "Data"}]


@mock.patch("requests.Session.get")
def test_get_user_info_empty_response(mock_get: mock.MagicMock) -> None:
    client = gss.superstaq_client.get_client(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_version="v0.3.0",
        api_key="to_my_heart",
        cq_token="cq-token",
    )
    mock_get.return_value.json.return_value = {}

    with pytest.raises(
        gss.SuperstaqServerException,
        match=("Something went wrong. The server has returned an empty response."),
    ):
        client.get_user_info()

    mock_get.assert_called_once_with(
        f"http://example.com/{API_VERSION}/user_info",
        headers=EXPECTED_HEADERS,
        verify=False,
    )
