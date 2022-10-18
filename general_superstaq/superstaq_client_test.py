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
import contextlib
import io
import json
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
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    assert (
        str(client)
        == f"Client with host=http://example.com/{API_VERSION} and name=general-superstaq"
    )
    assert str(eval(repr(client))) == str(client)


def test_general_superstaq_exception_str() -> None:
    ex = gss.SuperstaQException("err", status_code=501)
    assert str(ex) == "Status code: 501, Message: 'err'"


def test_general_superstaq_not_found_exception_str() -> None:
    ex = gss.SuperstaQNotFoundException("err")
    assert str(ex) == "Status code: 404, Message: 'err'"


def test_superstaq_client_invalid_remote_host() -> None:
    for invalid_url in ("", "url", "http://", "ftp://", "http://"):
        with pytest.raises(AssertionError, match="not a valid url"):
            _ = gss.superstaq_client._SuperstaQClient(
                client_name="general-superstaq", remote_host=invalid_url, api_key="a"
            )
        with pytest.raises(AssertionError, match=invalid_url):
            _ = gss.superstaq_client._SuperstaQClient(
                client_name="general-superstaq", remote_host=invalid_url, api_key="a"
            )


def test_superstaq_client_invalid_api_version() -> None:
    with pytest.raises(AssertionError, match="are accepted"):
        _ = gss.superstaq_client._SuperstaQClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version="v0.0",
        )
    with pytest.raises(AssertionError, match="0.0"):
        _ = gss.superstaq_client._SuperstaQClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            api_version="v0.0",
        )


def test_superstaq_client_time_travel() -> None:
    with pytest.raises(AssertionError, match="time machine"):
        _ = gss.superstaq_client._SuperstaQClient(
            client_name="general-superstaq",
            remote_host="http://example.com",
            api_key="a",
            max_retry_seconds=-1,
        )


def test_superstaq_client_attributes() -> None:
    client = gss.superstaq_client._SuperstaQClient(
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


@mock.patch("requests.post")
def test_supertstaq_client_create_job(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {"foo": "bar"}

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.create_job(
        serialized_circuits={"Hello": "World"},
        repetitions=200,
        target="ss_example_qpu",
        method="dry-run",
        options={"ibmq_pulse": True},
    )
    assert response == {"foo": "bar"}

    expected_json = {
        "Hello": "World",
        "target": "ss_example_qpu",
        "shots": 200,
        "method": "dry-run",
        "options": json.dumps({"ibmq_pulse": True}),
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

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException, match="Not authorized"):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_found(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_found

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_retriable(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException, match="Status code: 501"):
        _ = client.create_job({"Hello": "World"}, target="ss_example_qpu")


@mock.patch("requests.post")
def test_superstaq_client_create_job_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = gss.superstaq_client._SuperstaQClient(
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
    client = gss.superstaq_client._SuperstaQClient(
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

    client = gss.superstaq_client._SuperstaQClient(
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

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException, match="Status code: 400"):
        _ = client.create_job(
            serialized_circuits={"Hello": "World"},
            repetitions=200,
            target="ss_example_qpu",
        )


@mock.patch("requests.get")
def test_superstaq_client_get_job(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {"foo": "bar"}
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_job(job_id="job_id")
    assert response == {"foo": "bar"}

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/job/job_id", headers=EXPECTED_HEADERS, verify=False
    )


@mock.patch("requests.get")
def test_superstaq_client_get_balance(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {"balance": 123.4567}
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_balance()
    assert response == {"balance": 123.4567}

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/balance", headers=EXPECTED_HEADERS, verify=False
    )


@mock.patch("requests.post")
def test_superstaq_client_ibmq_set_token(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    client.ibmq_set_token({"ibmq_token": "token"})

    expected_json = {"ibmq_token": "token"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/ibmq_token",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_resource_estimate(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
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
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    response = client.get_targets()
    assert response == targets

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/targets", headers=EXPECTED_HEADERS, verify=False
    )


@mock.patch("requests.get")
def test_superstaq_client_get_job_unauthorized(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.unauthorized

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException, match="Not authorized"):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_not_found(mock_get: mock.MagicMock) -> None:
    (mock_get.return_value).ok = False
    (mock_get.return_value).status_code = requests.codes.not_found

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_not_retriable(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.bad_request

    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    with pytest.raises(gss.SuperstaQException, match="Status code: 400"):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_retry(mock_get: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_get.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    _ = client.get_job("job_id")
    assert mock_get.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_aqt_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.aqt_compile({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/aqt_compile"


@mock.patch("requests.post")
def test_superstaq_client_qscout_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.qscout_compile({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/qscout_compile"


@mock.patch("requests.post")
def test_superstaq_client_cq_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.cq_compile({"Hello": "1", "World": "2"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/cq_compile"


@mock.patch("requests.post")
def test_superstaq_client_ibmq_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.ibmq_compile(
        {"Hello": "1", "World": "2"},
    )

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/ibmq_compile"


@mock.patch("requests.post")
def test_superstaq_client_neutral_atom_compile(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.neutral_atom_compile(
        {"Hello": "1", "World": "2"},
    )

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/neutral_atom_compile"


@mock.patch("requests.post")
def test_superstaq_client_submit_qubo(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    target = "ss_example_qpu"
    repetitions = 10
    client.submit_qubo(example_qubo, target, repetitions=repetitions)

    expected_json = {
        "qubo": [
            {"keys": ["0"], "value": 1.0},
            {"keys": ["1"], "value": 1.0},
            {"keys": ["0", "1"], "value": -2.0},
        ],
        "target": target,
        "shots": repetitions,
    }

    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/qubo",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_find_min_vol_portfolio(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.find_min_vol_portfolio(
        {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "desired_return": 8}
    )

    expected_json = {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "desired_return": 8}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/minvol",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_find_max_pseudo_sharpe_ratio(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.find_max_pseudo_sharpe_ratio({"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "k": 0.5})

    expected_json = {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "k": 0.5}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/maxsharpe",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_tsp(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.tsp({"locs": ["Chicago", "St Louis", "St Paul"]})

    expected_json = {"locs": ["Chicago", "St Louis", "St Paul"]}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/tsp",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_warehouse(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    client.warehouse(
        {
            "k": 1,
            "possible_warehouses": ["Chicago", "San Francisco"],
            "customers": ["Rockford", "Aurora"],
        }
    )

    expected_json = {
        "k": 1,
        "possible_warehouses": ["Chicago", "San Francisco"],
        "customers": ["Rockford", "Aurora"],
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/warehouse",
        headers=EXPECTED_HEADERS,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_aqt_upload_configs(mock_post: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
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
    client = gss.superstaq_client._SuperstaQClient(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )

    assert client.aqt_get_configs() == expected_json
