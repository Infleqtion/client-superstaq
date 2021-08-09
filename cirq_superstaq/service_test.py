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

import os
from unittest import mock

import cirq
import pytest
import sympy

import cirq_superstaq


def test_service_run() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {
        "job_id": "job_id",
        "status": "ready",
    }
    mock_client.get_job.return_value = {
        "data": {"histogram": {"11": 1}},
        "job_id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "data": {"counts": {"0x3": 1}},
                "meas_level": 2,
                "seed_simulator": 775709958,
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "simulator",
    }

    service._client = mock_client

    a = sympy.Symbol("a")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X ** a)(q), cirq.measure(q, key="a"))
    params = cirq.ParamResolver({"a": 0.5})
    result = service.run(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        name="bacon",
        param_resolver=params,
    )
    assert result == {"11": 1}


def test_service_get_job() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    job_dict = {"job_id": "job_id", "status": "ready"}
    mock_client.get_job.return_value = job_dict
    service._client = mock_client

    job = service.get_job("job_id")
    assert job.job_id() == "job_id"
    mock_client.get_job.assert_called_with(job_id="job_id")


def test_service_create_job() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {"job_id": "job_id", "status": "ready"}
    mock_client.get_job.return_value = {"job_id": "job_id", "status": "completed"}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    job = service.create_job(circuit=circuit, repetitions=100, target="qpu")
    assert job.status() == "completed"
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "qpu"


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.aqt_compile",
    return_value={"compiled_circuit": cirq.to_json(cirq.Circuit())},
)
def test_service_aqt_compile(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    expected = cirq_superstaq.aqt.AQTCompilerOutput(cirq.Circuit())
    assert service.aqt_compile(cirq.Circuit()) == expected


def test_service_api_key_via_env() -> None:
    os.environ["SUPERSTAQ_API_KEY"] = "tomyheart"
    service = cirq_superstaq.Service(remote_host="http://example.com")
    assert service.api_key == "tomyheart"
    del os.environ["SUPERSTAQ_API_KEY"]


def test_service_remote_host_via_env() -> None:
    os.environ["SUPERSTAQ_REMOTE_HOST"] = "http://example.com"
    service = cirq_superstaq.Service(api_key="tomyheart")
    assert service.remote_host == "http://example.com"
    del os.environ["SUPERSTAQ_REMOTE_HOST"]


def test_service_no_param_or_env_variable() -> None:
    with pytest.raises(EnvironmentError):
        _ = cirq_superstaq.Service(remote_host="http://example.com")


def test_service_no_url_default() -> None:
    service = cirq_superstaq.Service(api_key="tomyheart")
    assert service.remote_host == cirq_superstaq.API_URL
