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
import contextlib
import io
import json
import os
from unittest import mock

import pytest
import qubovert as qv
import requests

import general_superstaq as gss

API_VERSION = gss.API_VERSION
EXPECTED_HEADERS = {
    "Authorization": "to_my_heart",
    "Content-Type": "application/json",
    "X-Client-Version": API_VERSION,
    "X-Client-Name": "general-superstaq",
}


def test_superstaq_client_str_and_repr() -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    assert (
        str(client)
        == f"Client with host=http://example.com/{API_VERSION} and name=general-superstaq"
    )
    assert str(eval(repr(client))) == str(client)


def test_superstaq_client_args() -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        cq_token="cq_token",
        ibmq_channel="ibm_quantum",
        ibmq_instance="instance",
        ibmq_token="ibmq_token",
    )
    assert client.client_kwargs == dict(
        cq_token="cq_token",
        ibmq_channel="ibm_quantum",
        ibmq_instance="instance",
        ibmq_token="ibmq_token",
    )

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


def test_warning_from_server() -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    warning = {"message": "WARNING!", "category": "SuperstaqWarning"}

    with mock.patch("requests.get", ok=True) as mock_request:
        mock_request.return_value.json = lambda: {"abc": 123, "warnings": [warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert client.get_request("/endpoint") == {"abc": 123}

    with mock.patch("requests.post", ok=True) as mock_request:
        mock_request.return_value.json = lambda: {"abc": 123, "warnings": [warning, warning]}
        with pytest.warns(gss.SuperstaqWarning, match="WARNING!"):
            assert client.post_request("/endpoint", {}) == {"abc": 123}


@pytest.mark.parametrize("invalid_url", ("url", "http://", "ftp://", "http://"))
def test_superstaq_client_invalid_remote_host(invalid_url: str) -> None:
    with pytest.raises(AssertionError, match="not a valid url"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq", remote_host=invalid_url, api_key="a"
        )
    with pytest.raises(AssertionError, match=invalid_url):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq", remote_host=invalid_url, api_key="a"
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


def test_superstaq_client_time_travel() -> None:
    with pytest.raises(AssertionError, match="time machine"):
        _ = gss.superstaq_client._SuperstaqClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            max_retry_seconds=-1,
        )


def test_superstaq_client_attributes() -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        max_retry_seconds=10,
        verbose=True,
    )
    assert client.url == f"http://example.com/{API_VERSION}"
    assert client.headers == EXPECTED_HEADERS
    assert client.max_retry_seconds == 10
    assert client.verbose


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient._accept_terms_of_use")
@mock.patch("requests.get")
def test_superstaq_client_needs_accept_terms_of_use(
    mock_get: mock.MagicMock,
    mock_accept_terms_of_use: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

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

    fake_authorized_get_response = mock.MagicMock(ok=True)
    mock_get.side_effect = [fake_get_response, fake_authorized_get_response]
    mock_accept_terms_of_use.return_value = "Accepted. You can now continue using Superstaq."
    with mock.patch("builtins.input"):
        client.get_balance()
        assert capsys.readouterr().out == "Accepted. You can now continue using Superstaq.\n"


@mock.patch("requests.post")
def test_superstaq_client_validate_email_error(
    mock_post: mock.MagicMock,
) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized
    mock_post.return_value.json.return_value = "You must validate your registered email."

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(
        gss.SuperstaqServerException, match="You must validate your registered email."
    ):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_supertstaq_client_create_job(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.json.return_value = {"foo": "bar"}

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    response = client.create_job(
        serialized_circuits={"Hello": "World"},
        repetitions=200,
        target="ss_example_qpu",
        method="dry-run",
        qiskit_pulse=True,
        cq_token={"@type": "RefreshFlowState", "access_token": "123"},
    )
    assert response == {"foo": "bar"}

    expected_json = {
        "Hello": "World",
        "target": "ss_example_qpu",
        "shots": 200,
        "method": "dry-run",
        "options": json.dumps(
            {"qiskit_pulse": True, "cq_token": {"@type": "RefreshFlowState", "access_token": "123"}}
        ),
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/jobs",
        json=expected_json,
        headers=EXPECTED_HEADERS,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_create_job_unauthorized(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_found(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_found

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_retriable(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Status code: 501"):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        verbose=True,
    )
    test_stdout = io.StringIO()
    with contextlib.redirect_stdout(test_stdout):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")
    assert test_stdout.getvalue().strip() == "Waiting 0.1 seconds before retrying."
    assert mock_post.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_create_job_retry_request_error(mock_post: mock.MagicMock) -> None:
    response2 = mock.MagicMock()
    mock_post.side_effect = [requests.exceptions.ConnectionError(), response2]
    response2.ok = True
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")
    assert mock_post.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_create_job_timeout(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.service_unavailable

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        max_retry_seconds=0.2,
    )
    with pytest.raises(TimeoutError):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_json(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.bad_request
    mock_post.return_value.json.return_value = {"message": "foo bar"}

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.create_job(
            serialized_circuits={"Hello": "World"},
            repetitions=200,
            target="ss_example_qpu",
        )


@mock.patch("requests.post")
def test_superstaq_client_fetch_jobs(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"my_id": {"foo": "bar"}}
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.fetch_jobs(job_ids=["job_id"], cq_token={"access_token": "token"})
    assert response == {"my_id": {"foo": "bar"}}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/fetch_jobs",
        json={
            "job_ids": ["job_id"],
            "options": '{"cq_token": {"access_token": "token"}}',
        },
        headers=EXPECTED_HEADERS,
        verify=False,
    )


@mock.patch("requests.get")
def test_superstaq_client_get_balance(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {"balance": 123.4567}
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_balance()
    assert response == {"balance": 123.4567}

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/balance", headers=EXPECTED_HEADERS, verify=False
    )


@mock.patch("requests.get")
def test_superstaq_client_get_version(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.headers = {"superstaq_version": "1.2.3"}
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_superstaq_version()
    assert response == {"superstaq_version": "1.2.3"}

    mock_get.assert_called_with(f"http://example.com/{API_VERSION}")


@mock.patch("requests.post")
def test_add_new_user(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    client.add_new_user({"Marie Curie": "mc@gmail.com"})

    expected_json = {"Marie Curie": "mc@gmail.com"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/add_new_user",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_update_user_balance(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    client.update_user_balance({"email": "mc@gmail.com", "balance": 5.00})

    expected_json = {"email": "mc@gmail.com", "balance": 5.00}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/update_user_balance",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_update_user_role(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    client.update_user_role({"email": "mc@gmail.com", "role": 5})

    expected_json = {"email": "mc@gmail.com", "role": 5}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/update_user_role",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_resource_estimate(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    client.resource_estimate({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/resource_estimate"


@mock.patch("requests.get")
def test_superstaq_client_get_targets(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    targets = {
        "superstaq_targets:": {
            "compile-and-run": [
                "ibmq_qasm_simulator",
                "ibmq_armonk_qpu",
                "ibmq_santiago_qpu",
                "ibmq_bogota_qpu",
                "ibmq_lima_qpu",
                "ibmq_belem_qpu",
                "ibmq_quito_qpu",
                "ibmq_statevector_simulator",
                "ibmq_mps_simulator",
                "ibmq_extended-stabilizer_simulator",
                "ibmq_stabilizer_simulator",
                "ibmq_manila_qpu",
                "aws_dm1_simulator",
                "aws_sv1_simulator",
                "d-wave_advantage-system4.1_qpu",
                "d-wave_dw-2000q-6_qpu",
                "aws_tn1_simulator",
                "rigetti_aspen-9_qpu",
                "d-wave_advantage-system1.1_qpu",
                "ionq_ion_qpu",
            ],
            "compile-only": ["aqt_keysight_qpu", "sandia_qscout_qpu"],
        }
    }
    mock_get.return_value.json.return_value = targets
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_targets()
    assert response == targets

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/targets", headers=EXPECTED_HEADERS, verify=False
    )


@mock.patch("requests.post")
def test_superstaq_client_get_job_unauthorized(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Not authorized"):
        _ = client.get_job("job_id")


@mock.patch("requests.post")
def test_superstaq_client_get_job_not_found(mock_post: mock.MagicMock) -> None:
    (mock_post.return_value).ok = False
    (mock_post.return_value).status_code = requests.codes.not_found

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException):
        _ = client.get_job("job_id")


@mock.patch("requests.post")
def test_superstaq_client_get_job_not_retriable(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.bad_request

    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaqServerException, match="Status code: 400"):
        _ = client.get_job("job_id")


@mock.patch("requests.post")
def test_superstaq_client_get_job_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    _ = client.get_job("job_id")
    assert mock_post.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_aqt_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.aqt_compile({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/aqt_compile"


@mock.patch("requests.post")
def test_superstaq_client_qscout_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.qscout_compile({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/qscout_compile"


@mock.patch("requests.post")
def test_superstaq_client_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.compile(
        {"Hello": "1", "World": "2"},
    )

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/compile"


@mock.patch("requests.post")
def test_superstaq_client_submit_qubo(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    target = "ss_example_qpu"
    repetitions = 10
    client.submit_qubo(
        example_qubo, target, repetitions=repetitions, method="dry-run", max_solutions=1
    )

    expected_json = {
        "qubo": [((0,), 1.0), ((1,), 1.0), ((0, 1), -2.0)],
        "target": target,
        "shots": repetitions,
        "method": "dry-run",
        "max_solutions": 1,
    }

    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/qubo",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_supercheq(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.supercheq([[0]], 1, 1, "cirq_circuits")

    expected_json = {
        "files": [[0]],
        "num_qubits": 1,
        "depth": 1,
        "circuit_return_type": "cirq_circuits",
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/supercheq",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_aces(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq", remote_host="http://example.com", api_key="to_my_heart"
    )
    client.submit_aces(
        target="ss_unconstrained_simulator",
        qubits=[0, 1],
        shots=100,
        num_circuits=10,
        mirror_depth=6,
        extra_depth=4,
        method="dry-run",
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
        "noise": {"type": "symmetric_depolarize", "params": (0.01,)},
        "tag": "test-tag",
        "lifespan": 10,
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/aces",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )

    client.process_aces("id")
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/aces_fetch",
        headers=EXPECTED_HEADERS,
        json={"job_id": "id"},
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_dfe(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
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
        f"http://example.com/{API_VERSION}/dfe_post",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )

    client.process_dfe(["id1", "id2"])
    expected_json = {"job_id_1": "id1", "job_id_2": "id2"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/dfe_fetch",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )

    with pytest.raises(ValueError, match="must contain exactly two job ids"):
        client.process_dfe(["1", "2", "3"])


@mock.patch("requests.post")
def test_superstaq_client_aqt_upload_configs(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
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


@mock.patch("requests.get")
def test_superstaq_client_aqt_get_configs(mock_get: mock.MagicMock) -> None:
    expected_json = {"pulses": "Hello", "variables": "World"}

    mock_get.return_value.json.return_value = expected_json
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    assert client.aqt_get_configs() == expected_json


@mock.patch("requests.post")
def test_superstaq_client_target_info(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.target_info("ss_example_qpu")

    expected_json = {"target": "ss_example_qpu"}

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
