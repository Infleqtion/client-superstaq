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
import uuid
from typing import Any
from unittest import mock

import pytest
import requests
from pytest import FixtureRequest

import general_superstaq as gss
from general_superstaq.testing import RETURNED_TARGETS, TARGET_LIST

# API_VERSION = gss.API_VERSION
EXPECTED_HEADERS = {
    "v0.2.0": {
        "Authorization": "to_my_heart",
        "Content-Type": "application/json",
        "X-Client-Version": "v0.2.0",
        "X-Client-Name": "general-superstaq",
    },
    "v0.3.0": {
        "Authorization": "to_my_heart",
        "Content-Type": "application/json",
        "X-Client-Version": "v0.3.0",
        "X-Client-Name": "general-superstaq",
    },
}


@pytest.fixture
def client_v2():
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version="v0.2.0",
    )
    return client


@pytest.fixture
def client_v3():
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version="v0.3.0",
    )
    return client


# @pytest.fixture
# def client_v2_token():
#     client = gss.superstaq_client._SuperstaqClient(
#         client_name="general-superstaq",
#         remote_host="http://example.com",
#         api_key="to_my_heart",
#         api_version="v0.2.0",
#         cq_token="cq-token",
#     )
#     return client


# @pytest.fixture
# def client_v3():
#     client = gss.superstaq_client._SuperstaqClient(
#         client_name="general-superstaq",
#         remote_host="http://example.com",
#         api_key="to_my_heart",
#         api_version="v0.3.0",
#         cq_token="cq-token",
#     )
#     return client


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
def test_superstaq_client_str_and_repr(client_name: str, request: FixtureRequest) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version
    assert (
        str(client)
        == f"Client version {api_version} with host=http://example.com/{api_version} and name=general-superstaq"
    )
    assert str(eval(repr(client))) == str(client)


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
def test_superstaq_client_args(api_version: str) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq_token",
        ibmq_channel="ibm_quantum",
        ibmq_instance="instance",
        ibmq_token="ibmq_token",
    )
    assert client.client_kwargs == {
        "cq_token": "cq_token",
        "ibmq_channel": "ibm_quantum",
        "ibmq_instance": "instance",
        "ibmq_token": "ibmq_token",
    }

    with pytest.raises(ValueError, match="must be either 'ibm_cloud' or 'ibm_quantum'"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="to_my_heart",
            ibmq_channel="foo",
        )


def test_general_superstaq_exception_str() -> None:
    ex = gss.SuperstaqServerException("err.", status_code=501)
    assert str(ex) == "err. (Status code: 501)"


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
def test_warning_from_server(client_name: str, request: FixtureRequest) -> None:
    client = request.getfixturevalue(client_name)

    warning = {"message": "WARNING!", "category": "SuperstaqWarning"}

    with mock.patch("requests.Session.get", ok=True) as mock_get:
        mock_get.return_value.json = lambda: {"abc": 123, "warnings": [warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert client.get_request("/endpoint") == {"abc": 123}

    with mock.patch("requests.Session.post", ok=True) as mock_post:
        mock_post.return_value.json = lambda: {"abc": 123, "warnings": [warning, warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert client.post_request("/endpoint", {}) == {"abc": 123}


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
@pytest.mark.parametrize("invalid_url", ("url", "http://", "ftp://", "http://"))
def test_superstaq_client_invalid_remote_host(api_version: str, invalid_url: str) -> None:
    with pytest.raises(AssertionError, match="not a valid url"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host=invalid_url,
            api_key="a",
            api_version=api_version,
        )
    with pytest.raises(AssertionError, match=invalid_url):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host=invalid_url,
            api_key="a",
            api_version=api_version,
        )


def test_superstaq_client_invalid_api_version() -> None:
    with pytest.raises(AssertionError, match="are accepted"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version="v0.0",
        )
    with pytest.raises(AssertionError, match="0.0"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version="v0.0",
        )


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
def test_superstaq_client_time_travel(api_version: str) -> None:
    with pytest.raises(AssertionError, match="time machine"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version=api_version,
            max_retry_seconds=-1,
        )


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
def test_superstaq_client_attributes(api_version: str) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        max_retry_seconds=10,
        verbose=True,
    )
    assert client.url == f"http://example.com/{api_version}"
    assert client.headers == EXPECTED_HEADERS[api_version]
    assert client.max_retry_seconds == 10
    assert client.verbose
    assert client.api_version == api_version


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("general_superstaq.superstaq_client._SuperstaqClient._accept_terms_of_use")
@mock.patch("requests.Session.get")
def test_superstaq_client_needs_accept_terms_of_use(
    mock_get: mock.MagicMock,
    mock_accept_terms_of_use: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = request.getfixturevalue(client_name)

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
            client.get_balance()
        assert capsys.readouterr().out == "YES response required to proceed\n"

    fake_authorized_get_response = mock.MagicMock()
    fake_authorized_get_response.ok = True
    fake_authorized_get_response.json.return_value = {"email": "test@email.com", "balance": 1.234}
    mock_get.side_effect = [fake_get_response, fake_authorized_get_response]
    mock_accept_terms_of_use.return_value = "Accepted. You can now continue using Superstaq."
    with mock.patch("builtins.input"):
        client.get_balance()
        assert capsys.readouterr().out == "Accepted. You can now continue using Superstaq.\n"


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_validate_email_error(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized
    mock_post.return_value.json.return_value = "You must validate your registered email."

    with pytest.raises(
        gss.SuperstaqServerException, match="You must validate your registered email."
    ):
        _ = client.create_job({"cirq_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
def test_superstaq_client_use_stored_ibmq_credential(api_version: str) -> None:
    credentials = {"token": "ibmq_token", "instance": "instance", "channel": "ibm_quantum"}
    with mock.patch(
        "general_superstaq.superstaq_client.read_ibm_credentials", return_value=credentials
    ):
        client = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="to_my_heart",
            api_version=api_version,
            cq_token="cq_token",
            use_stored_ibmq_credentials=True,
        )
        assert client.client_kwargs == {
            "cq_token": "cq_token",
            "ibmq_channel": "ibm_quantum",
            "ibmq_instance": "instance",
            "ibmq_token": "ibmq_token",
        }


@pytest.mark.parametrize("target", ("ss_example_qpu", "ss_example_simulator"))
@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.post")
def test_supertstaq_client_create_job(
    mock_post: mock.MagicMock,
    client_name: str,
    target: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.json.return_value = {"job_id": job_id, "num_circuits": 1}

    response = client.create_job(
        serialized_circuits={"qiskit_circuits": "World"},
        repetitions=200,
        target=target,
        method="dry-run",
        cq_token={"@type": "RefreshFlowState", "access_token": "123"},
    )
    assert response == {"job_id": job_id, "num_circuits": 1}

    if api_version == "v.0.2.0":
        expected_json = {
            "qiskit_circuits": "World",
            "target": target,
            "shots": 200,
            "method": "dry-run",
            "options": json.dumps(
                {"cq_token": {"@type": "RefreshFlowState", "access_token": "123"}}
            ),
        }
        mock_post.assert_called_with(
            "http://example.com/v0.2.0/jobs",
            json=expected_json,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
        )
    elif api_version == "v.0.3.0":
        job_type = "submit"
        if target.endswith("_simulator"):
            job_type = "simulate"
        expected_json = {
            "job_type": job_type,
            "target": target,
            "circuits": "World",
            "circuit_type": "qiskit",
            "verbatim": False,
            "shots": 200,
            "dry_run": True,
            "sim_method": None,
            "priority": 0,
            "optioins_dict": {},
            "tags": [],
        }
        mock_post.assert_called_with(
            "http://example.com/v0.3.0/client/jobs",
            json=expected_json,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
        )


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_unauthorized(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.create_job({"cirq_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_not_found(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_found

    with pytest.raises(gss.SuperstaqServerException):
        _ = client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_not_retriable(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
    )
    with pytest.raises(gss.SuperstaqServerException, match="Status code: 501"):
        _ = client.create_job({"cirq_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("api_version, job_id", [("v0.2.0", "id"), ("v0.3.0", uuid.UUID(int=0))])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_retry(
    mock_post: mock.MagicMock, api_version: str, job_id: str | uuid.UUID
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        verbose=True,
    )

    response1 = mock.MagicMock(ok=False, status_code=requests.codes.service_unavailable)
    response1.json.return_value = {"job_id": job_id, "num_circuits": 1}
    response2 = mock.MagicMock(ok=True)
    response2.json.return_value = {"job_id": job_id, "num_circuits": 1}
    mock_post.side_effect = [response1, response2]

    test_stdout = io.StringIO()
    with contextlib.redirect_stdout(test_stdout):
        _ = client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")
    assert test_stdout.getvalue().strip() == "Waiting 0.1 seconds before retrying."
    assert mock_post.call_count == 2


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_retry_request_error(
    mock_post: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    response2 = mock.MagicMock()
    mock_post.side_effect = [requests.exceptions.ConnectionError(), response2]
    response2.ok = True
    response2.json.return_value = {"job_id": job_id, "num_circuits": 1}

    _ = client.create_job({"cirq_circuits": "World"}, target="ss_example_qpu")
    assert mock_post.call_count == 2


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_invalid_json(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    response = requests.Response()
    response.status_code = requests.codes.not_implemented
    response._content = b"invalid/json"
    mock_post.return_value = response

    with mock.patch("requests.Session.post", return_value=response):
        with pytest.raises(gss.SuperstaqServerException, match="invalid/json"):
            _ = client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_dont_retry_on_timeout(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    response = requests.Response()
    response.status_code = requests.codes.gateway_timeout
    response._content = b"invalid/json"
    mock_post.return_value = response

    with mock.patch("requests.Session.post", return_value=response):
        with pytest.raises(gss.SuperstaqServerException, match="timed out"):
            _ = client.create_job({"cirq_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_timeout(mock_post: mock.MagicMock, api_version: str) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        max_retry_seconds=0.2,
    )

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.service_unavailable

    with pytest.raises(TimeoutError):
        _ = client.create_job({"qiskit_circuits": "World"}, target="ss_example_qpu")


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.post")
def test_superstaq_client_create_job_json(
    mock_post: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.bad_request
    mock_post.return_value.json.return_value = {
        "message": "foo bar",
        "job_id": job_id,
        "num_circuits": 1,
    }

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.create_job(
            serialized_circuits={"cirq_circuits": "World"},
            repetitions=200,
            target="ss_example_qpu",
        )


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"my_id": {"foo": "bar"}}
        response = client.fetch_jobs(job_ids=["job_id"], cq_token={"access_token": "token"})
        assert response == {"my_id": {"foo": "bar"}}

        mock_post.assert_called_with(
            f"http://example.com/{api_version}/fetch_jobs",
            json={
                "job_ids": ["job_id"],
                "options": '{"cq_token": {"access_token": "token"}}',
            },
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
        )
    else:
        expected_result = {
            uuid.UUID(int=0): {
                "job_type": "submit",
                "statuses": ["completed"],
                "status_messages": [None],
                "user_email": "test@email.com",
                "target": "ss_example_qpu",
                "provider_id": ["ss"],
                "num_circuits": 1,
                "compiled_circuits": ["compiled world"],
                "input_circuits": ["world"],
                "circuit_type": "cirq",
                "counts": [{"count": 200}],
                "results_dicts": [None],
                "shots": [200],
                "dry_run": True,
                "submission_timestamp": datetime.datetime(1, 1, 1),
                "last_updated_timestamp": [None],
                "initial_logical_to_physicals": [{0: 0}],
                "final_logical_to_physicals": [{0: 0}],
                "logical_qubits": ["0"],
                "physical_qubits": ["0"],
            }
        }
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = expected_result
        response = client.fetch_jobs(job_ids=[uuid.UUID(int=0)], cq_token={"access_token": "token"})
        assert response == expected_result

        mock_get.assert_called_with(
            f"http://example.com/{api_version}/client/job?job_id={uuid.UUID(int=0)}",
            headers=dict(EXPECTED_HEADERS[api_version], cq_token='{"access_token": "token"}'),
            verify=False,
        )


@pytest.mark.parametrize(
    "client_name, endpoint",
    [
        ("client_v2", "http://example.com/v0.2.0/balance"),
        ("client_v3", "http://example.com/v0.3.0/client/balance"),
    ],
)
@mock.patch("requests.Session.get")
def test_superstaq_client_get_balance(
    mock_get: mock.MagicMock,
    client_name: str,
    endpoint: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"balance": 123.4567}
        response = client.get_balance()
    else:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"balance": 123.4567, "email": "test@email.com"}
        response = client.get_balance()

    mock_get.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        verify=False,
    )
    assert response == {"balance": 123.4567}


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.get")
def test_superstaq_client_get_version(
    mock_get: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    mock_get.return_value.ok = True
    mock_get.return_value.headers = {"superstaq_version": "1.2.3"}

    response = client.get_superstaq_version()

    assert response == {"superstaq_version": "1.2.3"}
    mock_get.assert_called_with(f"http://example.com/{api_version}")


@pytest.mark.parametrize(
    "client_name, endpoint",
    [
        ("client_v2", "http://example.com/v0.2.0/add_new_user"),
        ("client_v3", "http://example.com/v0.3.0/client/user"),
    ],
)
@mock.patch("requests.Session.post")
def test_add_new_user(
    mock_post: mock.MagicMock,
    client_name: str,
    endpoint: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        expected_json = {"Marie Curie": "mc@gmail.com"}

    else:
        expected_json = {"name": "Marie Curie", "email": "mc@email.com", "role": "genius"}

    client.add_new_user(expected_json)
    mock_post.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        json=expected_json,
        verify=False,
    )


@pytest.mark.parametrize(
    "client_name, endpoint, expected_json",
    [
        (
            "client_v2",
            "http://example.com/v0.2.0/update_user_balance",
            {"email": "mc@gmail.com", "balance": 5.00},
        ),
        (
            "client_v3",
            "http://example.com/v0.3.0/client/user/mc@gmail.com",
            {"balance": 5.00},
        ),
    ],
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_update_user_balance(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    endpoint: str,
    expected_json: dict[str, Any],
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        mock_call = mock_post

    else:
        mock_call = mock_put

    client.update_user_balance({"email": "mc@gmail.com", "balance": 5.00})
    mock_call.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        json=expected_json,
        verify=False,
    )


def test_update_user_balance_invalid_v3(client_v3) -> None:
    with pytest.raises(ValueError, match="user email"):
        client_v3.update_user_balance({"balance": 5.00})

    with pytest.raises(ValueError, match="new balance"):
        client_v3.update_user_balance({"email": "test@email.com"})


@pytest.mark.parametrize(
    "client_name, endpoint, role, expected_json",
    [
        (
            "client_v2",
            "http://example.com/v0.2.0/update_user_role",
            5,
            {"email": "mc@gmail.com", "role": 5},
        ),
        (
            "client_v3",
            "http://example.com/v0.3.0/client/user/mc@gmail.com",
            "genius",
            {"role": "genius"},
        ),
    ],
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_update_user_role(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    endpoint: str,
    role: int | str,
    expected_json: dict[str, Any],
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        mock_call = mock_post

    else:
        mock_call = mock_put

    client.update_user_role({"email": "mc@gmail.com", "role": role})
    mock_call.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        json=expected_json,
        verify=False,
    )


def test_update_user_role_invalid_v3(client_v3) -> None:
    with pytest.raises(ValueError, match="user email"):
        client_v3.update_user_role({"role": "genius"})

    with pytest.raises(ValueError, match="new role"):
        client_v3.update_user_role({"email": "test@email.com"})


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_resource_estimate(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.resource_estimate({"Hello": "1", "World": "2"})
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == f"http://example.com/{api_version}/resource_estimate"
    else:
        with pytest.raises(NotImplementedError, match="resource_estimate is not implemented"):
            client.resource_estimate({"Hello": "1", "World": "2"})


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/targets"),
        ("v0.3.0", "http://example.com/v0.3.0/client/targets"),
    ],
)
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_get_targets(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
    )
    token_client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        ibmq_token="token",
    )
    if api_version == "v0.2.0":
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"superstaq_targets": TARGET_LIST}
        response = client.get_targets()
        assert response == RETURNED_TARGETS
        mock_post.assert_called_once_with(
            endpoint,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
            json={},
        )

        response = client.get_targets(simulator=True)
        mock_post.assert_called_with(
            endpoint,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
            json={"simulator": True},
        )

        response = token_client.get_targets(simulator=True)
        mock_post.assert_called_with(
            endpoint,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
            json={"simulator": True, "options": json.dumps({"ibmq_token": "token"})},
        )
    else:
        target_list_v3 = [
            dict(target_name=name, simulator=True, accessible=False, **properties)
            for name, properties in TARGET_LIST.items()
        ]
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = target_list_v3
        response = client.get_targets()
        assert response == RETURNED_TARGETS
        mock_get.assert_called_once_with(
            endpoint,
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
        )

        response = client.get_targets(simulator=True)
        mock_get.assert_called_with(
            endpoint + "?simulator=True",
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
        )

        response = token_client.get_targets(simulator=True)
        mock_get.assert_called_with(
            endpoint + "?simulator=True",
            headers={**EXPECTED_HEADERS[api_version], "ibmq_token": "token"},
            verify=False,
        )


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_get_my_targets(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    api_version: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        ibmq_token="token",
    )
    target = {
        "ss_unconstrained_simulator": {
            "supports_submit": True,
            "supports_submit_qubo": True,
            "supports_compile": True,
            "available": True,
            "retired": False,
            "accessible": True,
        }
    }
    if api_version == "v0.2.0":
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"superstaq_targets": target}
        response = client.get_my_targets()
        assert response == [
            gss.typing.Target(
                target="ss_unconstrained_simulator",
                **target["ss_unconstrained_simulator"],
            )
        ]
        mock_post.assert_called_once_with(
            f"http://example.com/{api_version}/targets",
            headers=EXPECTED_HEADERS[api_version],
            verify=False,
            json={"accessible": True, "options": json.dumps({"ibmq_token": "token"})},
        )
    else:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = [
            {
                "target_name": "ss_unconstrained_simulator",
                "simulator": True,
                **target["ss_unconstrained_simulator"],
            }
        ]
        response = client.get_my_targets()
        assert response == [
            gss.typing.Target(
                target="ss_unconstrained_simulator",
                **target["ss_unconstrained_simulator"],
            )
        ]
        mock_get.assert_called_once_with(
            f"http://example.com/{api_version}/client/targets?accessible=True",
            headers={**EXPECTED_HEADERS[api_version], "ibmq_token": "token"},
            verify=False,
        )


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs_unauthorized(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_get

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.unauthorized

    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.fetch_jobs([job_id])


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs_not_found(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_get

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.not_found

    with pytest.raises(gss.SuperstaqServerException):
        _ = client.fetch_jobs([job_id])


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs_not_retriable(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_get

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.bad_request

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.fetch_jobs([job_id])


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_fetch_jobs_retry(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    response1 = mock.MagicMock()
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable

    response2 = mock.MagicMock()
    response2.ok = True

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_get

    mock_call.side_effect = [response1, response2]
    _ = client.fetch_jobs([job_id])

    assert mock_call.call_count == 2


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_unauthorized(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_put

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.unauthorized

    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.cancel_jobs([job_id])


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_not_found(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_put

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.not_found

    with pytest.raises(gss.SuperstaqServerException):
        _ = client.cancel_jobs([job_id])


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_superstaq_client_get_cancel_jobs_retriable(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_put

    mock_call.return_value.ok = False
    mock_call.return_value.status_code = requests.codes.bad_request

    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.cancel_jobs([job_id], cq_token=1)


@pytest.mark.parametrize(
    "client_name, job_id", [("client_v2", "id"), ("client_v3", uuid.UUID(int=0))]
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_superstaq_client_cancel_jobs_retry(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    job_id: str | uuid.UUID,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    response1 = mock.MagicMock(ok=False, status_code=requests.codes.service_unavailable)
    response2 = mock.MagicMock(ok=True)

    if client.api_version == "v0.2.0":
        mock_call = mock_post

    else:
        response2.json.return_value = {
            "succeeded": [str(job_id)],
            "message": "",
        }
        mock_call = mock_put

    mock_call.side_effect = [response1, response2]
    _ = client.cancel_jobs([job_id])
    assert mock_call.call_count == 2


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_aqt_compile(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.aqt_compile({"Hello": "1", "World": "2"})
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == f"http://example.com/{api_version}/aqt_compile"
    else:
        with pytest.raises(DeprecationWarning, match="`aqt_compile` is deprecated"):
            client.aqt_compile({"Hello": "1", "World": "2"})


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_qscout_compile(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.qscout_compile({"Hello": "1", "World": "2"})
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == f"http://example.com/{api_version}/qscout_compile"
    else:
        with pytest.raises(DeprecationWarning, match="`qscout_compile` is deprecated"):
            client.qscout_compile({"Hello": "1", "World": "2"})


@mock.patch("requests.Session.post")
def test_superstaq_client_compile_v2(
    mock_post: mock.MagicMock,
    client_v2: gss.superstaq_client._SuperstaqClient,
) -> None:
    client_v2.compile(
        {"Hello": "1", "World": "2"},
    )

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{client_v2.api_version}/compile"


def test_superstaq_client_compile_v3_multi_circuit_types(
    client_v3: gss.superstaq_client._SuperstaqClient,
) -> None:
    with pytest.raises(RuntimeError, match="multiple circuit types"):
        client_v3.compile({"cirq_circuits": "Hello", "qiskit_circuits": "World"})


def test_superstaq_client_compile_v3__type_not_found(
    client_v3: gss.superstaq_client._SuperstaqClient,
) -> None:
    with pytest.raises(RuntimeError, match="No recognized circuits found."):
        client_v3.compile({"qasm_circuits": "Hello"})


@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_compile_v3_failed(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_v3: gss.superstaq_client._SuperstaqClient,
) -> None:
    job_id = uuid.UUID(int=0)
    job_data = {
        str(job_id): {
            "job_type": "compile",
            "statuses": ["failed"],
            "status_messages": [None],
            "user_email": "test@email.com",
            "target": "ss_example_qpu",
            "provider_id": ["ss"],
            "num_circuits": 1,
            "compiled_circuits": ["compiled world"],
            "input_circuits": ["world"],
            "circuit_type": "cirq",
            "counts": [{"count": 200}],
            "results_dicts": [None],
            "shots": [200],
            "dry_run": True,
            "submission_timestamp": datetime.datetime(1, 1, 1),
            "last_updated_timestamp": [None],
            "initial_logical_to_physicals": [{0: 0}],
            "final_logical_to_physicals": [{0: 0}],
            "logical_qubits": ["0"],
            "physical_qubits": ["0"],
        }
    }
    mock_post.return_value.json.return_value = {"job_id": job_id, "num_circuits": 1}
    mock_get.return_value.json.return_value = job_data

    with pytest.raises(gss.SuperstaqException, match=f"Check job ID {job_id} for further details."):
        _ = client_v3.compile({"cirq_circuits": "Hello", "target": "ss_example_qpu"})

    mock_post.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job",
        json={
            "job_type": "compile",
            "target": "ss_example_qpu",
            "circuits": "Hello",
            "circuit_type": "cirq",
            "verbatim": False,
            "shots": 0,
            "dry_run": False,
            "sim_method": None,
            "priority": 0,
            "options_dict": {},
            "tags": [],
        },
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )
    mock_get.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job?job_id={job_id}",
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )


@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_compile_v3(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_v3: gss.superstaq_client._SuperstaqClient,
) -> None:
    job_id = uuid.UUID(int=0)
    job_data = {
        str(job_id): {
            "job_type": "compile",
            "statuses": ["completed"],
            "status_messages": [None],
            "user_email": "test@email.com",
            "target": "ss_example_qpu",
            "provider_id": ["ss"],
            "num_circuits": 1,
            "compiled_circuits": ["compiled world"],
            "input_circuits": ["world"],
            "circuit_type": "cirq",
            "counts": [{"count": 200}],
            "results_dicts": [None],
            "shots": [200],
            "dry_run": True,
            "submission_timestamp": datetime.datetime(1, 1, 1),
            "last_updated_timestamp": [None],
            "initial_logical_to_physicals": [{0: 0}],
            "final_logical_to_physicals": [{0: 0}],
            "logical_qubits": ["0"],
            "physical_qubits": ["0"],
        }
    }
    mock_post.return_value.json.return_value = {"job_id": job_id, "num_circuits": 1}
    mock_get.return_value.json.return_value = job_data

    compilation_results = client_v3.compile({"cirq_circuits": "Hello", "target": "ss_example_qpu"})

    mock_post.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job",
        json={
            "job_type": "compile",
            "target": "ss_example_qpu",
            "circuits": "Hello",
            "circuit_type": "cirq",
            "verbatim": False,
            "shots": 0,
            "dry_run": False,
            "sim_method": None,
            "priority": 0,
            "options_dict": {},
            "tags": [],
        },
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )
    mock_get.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job?job_id={job_id}",
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )
    assert compilation_results == {
        "cirq_circuits": "[compiled world]",
        "initial_logical_to_physicals": "[{0: 0}]",
        "final_logical_to_physicals": "[{0: 0}]",
    }


@mock.patch("requests.Session.get")
@mock.patch("requests.Session.post")
def test_superstaq_client_compile_v3_with_wait(
    mock_post: mock.MagicMock,
    mock_get: mock.MagicMock,
    client_v3: gss.superstaq_client._SuperstaqClient,
) -> None:
    job_id = uuid.UUID(int=0)
    compiling_data = {
        "job_type": "compile",
        "statuses": ["compiling"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_example_qpu",
        "provider_id": ["ss"],
        "num_circuits": 1,
        "compiled_circuits": ["compiled world"],
        "input_circuits": ["world"],
        "circuit_type": "cirq",
        "counts": [{"count": 200}],
        "results_dicts": [None],
        "shots": [200],
        "dry_run": True,
        "submission_timestamp": datetime.datetime(1, 1, 1),
        "last_updated_timestamp": [None],
        "initial_logical_to_physicals": [{0: 0}],
        "final_logical_to_physicals": [{0: 0}],
        "logical_qubits": ["0"],
        "physical_qubits": ["0"],
    }
    
    mock_post.return_value.json.return_value = {"job_id": job_id, "num_circuits": 1}
    response1 = mock.MagicMock()
    response1.json.return_value = {str(job_id): compiling_data}
    completed_data = compiling_data.copy()
    completed_data["statuses"] = ["completed"] 
    response2 = mock.MagicMock()
    response2.json.return_value = {str(job_id): completed_data}
    mock_get.side_effect = [response1, response2]

    compilation_results = client_v3.compile({"cirq_circuits": "Hello", "target": "ss_example_qpu"})

    mock_post.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job",
        json={
            "job_type": "compile",
            "target": "ss_example_qpu",
            "circuits": "Hello",
            "circuit_type": "cirq",
            "verbatim": False,
            "shots": 0,
            "dry_run": False,
            "sim_method": None,
            "priority": 0,
            "options_dict": {},
            "tags": [],
        },
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )
    mock_get.assert_called_with(
        f"http://example.com/{client_v3.api_version}/client/job?job_id={job_id}",
        headers=EXPECTED_HEADERS[client_v3.api_version],
        verify=False,
    )
    assert mock_get.call_count == 2
    assert compilation_results == {
        "cirq_circuits": "[compiled world]",
        "initial_logical_to_physicals": "[{0: 0}]",
        "final_logical_to_physicals": "[{0: 0}]",
    }


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_submit_qubo(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    example_qubo = {
        ("a",): 2.0,
        ("a", "b"): 1.0,
        ("b", 0): -5,
        (): -3.0,
    }
    target = "ss_unconstrained_simulator"
    repetitions = 10

    if api_version == "v0.2.0":
        client.submit_qubo(
            example_qubo,
            target,
            method="qaoa",
            repetitions=repetitions,
            max_solutions=1,
            foo_kwarg=True,
        )

        expected_json = {
            "qubo": [(("a",), 2.0), (("a", "b"), 1.0), (("b", 0), -5), ((), -3.0)],
            "target": target,
            "shots": repetitions,
            "method": "qaoa",
            "max_solutions": 1,
            "dry_run": False,
            "options": json.dumps(
                {
                    "qaoa_depth": 1,
                    "rqaoa_cutoff": 0,
                    "random_seed": None,
                    "foo_kwarg": True,
                }
            ),
        }

        mock_post.assert_called_with(
            f"http://example.com/{api_version}/qubo",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )
    else:
        with pytest.raises(NotImplementedError, match="submit_qubo is not implemented"):
            client.submit_qubo(
                example_qubo,
                target,
                method="qaoa",
                repetitions=repetitions,
                max_solutions=1,
                foo_kwarg=True,
            )


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_supercheq(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.supercheq([[0]], 1, 1, "cirq_circuits")

        expected_json = {
            "files": [[0]],
            "num_qubits": 1,
            "depth": 1,
            "circuit_return_type": "cirq_circuits",
        }
        mock_post.assert_called_with(
            f"http://example.com/{api_version}/supercheq",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )
    else:
        with pytest.raises(NotImplementedError, match="supercheq is not implemented"):
            client.supercheq([[0]], 1, 1, "cirq_circuits")


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_aces(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.submit_aces(
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

        expected_json = {
            "target": "ss_unconstrained_simulator",
            "qubits": [0, 1],
            "shots": 100,
            "num_circuits": 10,
            "mirror_depth": 6,
            "extra_depth": 4,
            "method": "dry-run",
            "weights": [1, 2],
            "noise": {"type": "symmetric_depolarize", "params": (0.01,)},
            "tag": "test-tag",
            "lifespan": 10,
        }
        mock_post.assert_called_with(
            f"http://example.com/{api_version}/aces",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )

        client.process_aces("id")
        mock_post.assert_called_with(
            f"http://example.com/{api_version}/aces_fetch",
            headers=EXPECTED_HEADERS[api_version],
            json={"job_id": "id"},
            verify=False,
        )
    else:
        with pytest.raises(NotImplementedError, match="submit_aces is not implemented"):
            client.submit_aces(
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


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_cb(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.submit_cb(
            target="ss_unconstrained_simulator",
            shots=100,
            serialized_circuits={"circuits": "test_circuit_data"},
            n_channels=6,
            n_sequences=30,
            depths=[2, 4, 6],
            method="dry-run",
            noise={"type": "symmetric_depolarize", "params": (0.01,)},
        )

        expected_json = {
            "target": "ss_unconstrained_simulator",
            "shots": 100,
            "circuits": "test_circuit_data",
            "n_channels": 6,
            "n_sequences": 30,
            "depths": [2, 4, 6],
            "method": "dry-run",
            "noise": {"type": "symmetric_depolarize", "params": (0.01,)},
        }

        mock_post.assert_called_with(
            f"http://example.com/{api_version}/cb_submit",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )

        client.process_cb("id", counts="[{}]")
        mock_post.assert_called_with(
            f"http://example.com/{api_version}/cb_fetch",
            headers=EXPECTED_HEADERS[api_version],
            json={"job_id": "id", "counts": "[{}]"},
            verify=False,
        )
    else:
        with pytest.raises(NotImplementedError, match="submit_cb is not implemented"):
            client.submit_cb(
                target="ss_unconstrained_simulator",
                shots=100,
                serialized_circuits={"circuits": "test_circuit_data"},
                n_channels=6,
                n_sequences=30,
                depths=[2, 4, 6],
                method="dry-run",
                noise={"type": "symmetric_depolarize", "params": (0.01,)},
            )


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.post")
def test_superstaq_client_dfe(
    mock_post: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    api_version = client.api_version

    if api_version == "v0.2.0":
        client.submit_dfe(
            circuit_1={"Hello": "World"},
            target_1="ss_example_qpu",
            circuit_2={"Hello": "World"},
            target_2="ss_example_qpu",
            num_random_bases=5,
            shots=100,
            lifespan=10,
        )

        state = {
            "Hello": "World",
            "target": "ss_example_qpu",
        }
        expected_json = {
            "state_1": state,
            "state_2": state,
            "shots": 100,
            "n_bases": 5,
            "options": json.dumps({"lifespan": 10}),
        }

        mock_post.assert_called_with(
            f"http://example.com/{api_version}/dfe_post",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )

        client.process_dfe(["id1", "id2"])
        expected_json = {"job_id_1": "id1", "job_id_2": "id2"}
        mock_post.assert_called_with(
            f"http://example.com/{api_version}/dfe_fetch",
            headers=EXPECTED_HEADERS[api_version],
            json=expected_json,
            verify=False,
        )

        with pytest.raises(ValueError, match="must contain exactly two job ids"):
            client.process_dfe(["1", "2", "3"])
    else:
        with pytest.raises(NotImplementedError, match="submit_dfe is not implemented"):
            client.submit_dfe(
                circuit_1={"Hello": "World"},
                target_1="ss_example_qpu",
                circuit_2={"Hello": "World"},
                target_2="ss_example_qpu",
                num_random_bases=5,
                shots=100,
                lifespan=10,
            )


@pytest.mark.parametrize(
    "client_name, endpoint",
    [
        ("client_v2", "http://example.com/v0.2.0/aqt_configs"),
        ("client_v3", "http://example.com/v0.3.0/aqt_configs"),
    ],
)
@mock.patch("requests.Session.put")
@mock.patch("requests.Session.post")
def test_superstaq_client_aqt_upload_configs(
    mock_post: mock.MagicMock,
    mock_put: mock.MagicMock,
    client_name: str,
    endpoint: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)

    client.aqt_upload_configs({"pulses": "Hello", "variables": "World"})

    expected_json = {"pulses": "Hello", "variables": "World"}

    if client.api_version == "v0.2.0":
        mock_call = mock_post
    else:
        mock_call = mock_put

    mock_call.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[client.api_version],
        json=expected_json,
        verify=False,
    )


@pytest.mark.parametrize("client_name", ["client_v2", "client_v3"])
@mock.patch("requests.Session.get")
def test_superstaq_client_aqt_get_configs(
    mock_get: mock.MagicMock,
    client_name: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    expected_json = {"pulses": "Hello", "variables": "World"}

    mock_get.return_value.json.return_value = expected_json

    assert client.aqt_get_configs() == expected_json


@pytest.mark.parametrize(
    "client_name, endpoint",
    [
        ("client_v2", "http://example.com/v0.2.0/target_info"),
        ("client_v3", "http://example.com/v0.3.0/client/retrieve_target_info"),
    ],
)
@mock.patch("requests.Session.post")
def test_superstaq_client_target_info(
    mock_post: mock.MagicMock,
    client_name: str,
    endpoint: str,
    request: FixtureRequest,
) -> None:
    client = request.getfixturevalue(client_name)
    mock_post.return_value.ok.return_value = True
    mock_post.return_value.json.return_value = {"target_info": {"Hello": "World"}}

    if client.api_version == "v0.2.0":
        expected_json = {"target": "ss_example_qpu", "options": "{}"}
    else:
        expected_json = {"target": "ss_example_qpu", "options_dict": {}}

    client.target_info("ss_example_qpu")
    mock_post.assert_called_with(
        endpoint,
        headers=EXPECTED_HEADERS[client.api_version],
        json=expected_json,
        verify=False,
    )


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/target_info"),
        ("v0.3.0", "http://example.com/v0.3.0/client/retrieve_target_info"),
    ],
)
@mock.patch("requests.Session.post")
def test_superstaq_client_target_info_with_credentials(
    mock_post: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq-token",
    )

    mock_post.return_value.json.return_value = {"target_info": {"Hello": "World"}}
    client.target_info("ss_example_qpu")

    if api_version == "v0.2.0":
        expected_json = {
            "target": "ss_example_qpu",
            "options": json.dumps({"cq_token": "cq-token"}),
        }
        headers = EXPECTED_HEADERS[api_version]
    else:
        expected_json = {"target": "ss_example_qpu", "options_dict": {}}
        headers = {**EXPECTED_HEADERS[api_version], "cq_token": "cq-token"}
    mock_post.assert_called_with(
        endpoint,
        headers=headers,
        json=expected_json,
        verify=False,
    )


def test_read_ibm_credentials() -> None:
    credentials = {
        "default-ibm-quantum": {
            "token": "ibmq_token",
            "instance": "instance",
            "channel": "ibm_quantum",
            "is_default_account": "True",
        },
        "myAccount": {"token": "account_token", "channel": "ibm_cloud"},
    }
    one_none_default_account = {"myAccount": {"token": "account_token", "channel": "ibm_cloud"}}
    multiple_none_default_account = {
        "myAccount": {"token": "account_token", "channel": "ibm_cloud"},
        "otherAccount": {"token": "other_token", "channel": "ibm_quantum"},
    }

    bad_credentials = {"account": {"instance": "instance", "channel": "ibm_quantum"}}

    with mock.patch("pathlib.Path.is_file", return_value=True):
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(credentials))):
            # multiple accounts with a default marked
            assert gss.superstaq_client.read_ibm_credentials(None) == credentials.get(
                "default-ibm-quantum"
            )
            assert gss.superstaq_client.read_ibm_credentials("myAccount") == credentials.get(
                "myAccount"
            )
        # only one account
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=json.dumps(one_none_default_account))
        ):
            assert gss.superstaq_client.read_ibm_credentials(None) == one_none_default_account.get(
                "myAccount"
            )

        # fail because multiple accounts found with none marked as default
        with pytest.raises(
            ValueError, match="Multiple accounts found but none are marked as default."
        ):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=json.dumps(multiple_none_default_account))
            ):
                gss.superstaq_client.read_ibm_credentials(None)

        # fail because provided name is not an account in the config
        with pytest.raises(KeyError, match="No account credentials saved under the name 'bad_key'"):
            with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(credentials))):
                gss.superstaq_client.read_ibm_credentials("bad_key")

        # fail with missing token field in config
        with pytest.raises(
            KeyError, match="`token` and/or `channel` keys missing from credentials"
        ):
            with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(bad_credentials))):
                gss.superstaq_client.read_ibm_credentials(None)
    #
    # fail to find credentials file
    with pytest.raises(FileNotFoundError, match="The `qiskit-ibm.json` file was not found in"):
        with mock.patch("pathlib.Path.is_file", return_value=False):
            gss.superstaq_client.read_ibm_credentials(None)


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


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/user_info"),
        ("v0.3.0", "http://example.com/v0.3.0/client/user"),
    ],
)
@mock.patch("requests.Session.get")
def test_get_user_info(
    mock_get: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq-token",
    )

    if api_version == "v0.2.0":
        mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}
        data = [{"Some": "Data"}]
    else:
        data = [
            {
                "name": "Marie Curie",
                "email": "mc@email.com",
                "role": "genius",
                "balance": 1867.0,
                "token": "cq-token",
                "user_id": uuid.UUID(int=0),
            }
        ]
        mock_get.return_value.json.return_value = data

    user_info = client.get_user_info()
    mock_get.assert_called_once_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        verify=False,
    )
    assert user_info == data


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/user_info"),
        ("v0.3.0", "http://example.com/v0.3.0/client/user"),
    ],
)
@mock.patch("requests.Session.get")
def test_get_user_info_query(
    mock_get: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq-token",
    )

    if api_version == "v0.2.0":
        mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}
        data = [{"Some": "Data"}]
    else:
        data = [
            {
                "name": "Alice",
                "email": "al@email.com",
                "role": "genius",
                "balance": 1867.0,
                "token": "cq-token",
                "user_id": uuid.UUID(int=0),
            }
        ]
        mock_get.return_value.json.return_value = data

    user_info = client.get_user_info(name="Alice")
    mock_get.assert_called_once_with(
        endpoint + "?name=Alice",
        headers=EXPECTED_HEADERS[api_version],
        verify=False,
    )
    assert user_info == data


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/user_info"),
        ("v0.3.0", "http://example.com/v0.3.0/client/user"),
    ],
)
@mock.patch("requests.Session.get")
def test_get_user_info_query_composite(
    mock_get: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq-token",
    )

    if api_version == "v0.2.0":
        mock_get.return_value.json.return_value = {"example@email.com": {"Some": "Data"}}
        data = [{"Some": "Data"}]
        user_id = 42
        endpoint = endpoint + f"?name=Alice&id={user_id}"
        # user_info = client.get_user_info(user_id=user_id, name="Alice")

    else:
        data = [
            {
                "name": "Alice",
                "email": "al@email.com",
                "role": "genius",
                "balance": 1867.0,
                "token": "cq-token",
                "user_id": uuid.UUID(int=0),
            }
        ]
        mock_get.return_value.json.return_value = data
        user_id = uuid.UUID(int=0)
        endpoint = endpoint + f"?name=Alice&user_id={user_id}"

    user_info = client.get_user_info(user_id=user_id, name="Alice")

    mock_get.assert_called_once_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        verify=False,
    )
    assert user_info == data


@pytest.mark.parametrize(
    "api_version, endpoint",
    [
        ("v0.2.0", "http://example.com/v0.2.0/user_info"),
        ("v0.3.0", "http://example.com/v0.3.0/client/user"),
    ],
)
@mock.patch("requests.Session.get")
def test_get_user_info_empty_response(
    mock_get: mock.MagicMock,
    api_version: str,
    endpoint: str,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version=api_version,
        cq_token="cq-token",
    )
    mock_get.return_value.json.return_value = {}

    with pytest.raises(
        gss.SuperstaqServerException,
        match=("Something went wrong. The server has returned an empty response."),
    ):
        client.get_user_info()

    mock_get.assert_called_once_with(
        endpoint,
        headers=EXPECTED_HEADERS[api_version],
        verify=False,
    )
