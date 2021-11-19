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

import collections
import os
import textwrap
from unittest import mock

import applications_superstaq
import cirq
import pandas as pd
import pytest
import sympy

import cirq_superstaq


def test_counts_to_results() -> None:
    qubits = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        cirq.H(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0]),
        cirq.measure(qubits[1]),
    )
    result = cirq_superstaq.service.counts_to_results(
        collections.Counter({"01": 1, "11": 2}), circuit, cirq.ParamResolver({})
    )
    assert result.histogram(key="01") == collections.Counter({3: 2, 1: 1})

    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0]),
        cirq.measure(qubits[1]),
    )
    result = cirq_superstaq.service.counts_to_results(
        collections.Counter({"00": 50, "11": 50}), circuit, cirq.ParamResolver({})
    )
    assert result.histogram(key="01") == collections.Counter({0: 50, 3: 50})


def test_service_run_and_get_counts() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
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
    counts = service.get_counts(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        name="bacon",
        param_resolver=params,
    )
    assert counts == {"11": 1}

    result = service.run(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        name="bacon",
        param_resolver=params,
    )
    assert result.histogram(key="a") == collections.Counter({3: 1})


def test_service_sampler() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    service._client = mock_client
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }
    mock_client.get_job.return_value = {
        "data": {"histogram": {"0": 3, "1": 1}},
        "num_qubits": 1,
        "job_id": "my_id",
        "samples": {"0": 3, "1": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "ibmq_qasm_simulator",
    }

    sampler = service.sampler(target="ibmq_qasm_simulator")
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="a"))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=["a"], index=[0, 1, 2, 3], data=[[0], [0], [0], [1]])
    )
    mock_client.create_job.assert_called_once()


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
    mock_client.create_job.return_value = {"job_ids": ["job_id"], "status": "ready"}
    mock_client.get_job.return_value = {"job_id": "job_id", "status": "completed"}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    job = service.create_job(circuit=circuit, repetitions=100, target="qpu")
    assert job.status() == "completed"
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "qpu"


def test_service_get_balance() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    service._client = mock_client

    assert service.get_balance() == "$12,345.68"
    assert service.get_balance(pretty_output=False) == 12345.6789


def test_service_get_backends() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    backends = {
        "superstaq_backends": {
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
    mock_client.get_backends.return_value = backends
    service._client = mock_client

    assert service.get_backends() == backends["superstaq_backends"]


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.aqt_compile",
    return_value={
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits(cirq.Circuit()),
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]]]),
    },
)
def test_service_aqt_compile_single(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    out = service.aqt_compile(cirq.Circuit())
    assert out.circuit == cirq.Circuit()
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.aqt_compile",
    return_value={
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits(
            [cirq.Circuit(), cirq.Circuit()]
        ),
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]], [[]]]),
    },
)
def test_service_aqt_compile_multiple(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    out = service.aqt_compile([cirq.Circuit(), cirq.Circuit()])
    assert out.circuits == [cirq.Circuit(), cirq.Circuit()]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


@mock.patch("applications_superstaq.superstaq_client._SuperstaQClient.qscout_compile")
def test_service_qscout_compile_single(mock_qscout_compile: mock.MagicMock) -> None:

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    jaqal_program = textwrap.dedent(
        """\
                register allqubits[1]
                prepare_all
                R allqubits[0] -1.5707963267948966 1.5707963267948966
                Rz allqubits[0] -3.141592653589793
                measure_all
                """
    )

    mock_qscout_compile.return_value = {
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits(circuit),
        "jaqal_programs": [jaqal_program],
    }

    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    out = service.qscout_compile(circuit)
    assert out.circuit == circuit
    assert out.jaqal_programs == jaqal_program


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.ibmq_compile",
    return_value={"pulses": applications_superstaq.converters.serialize([mock.DEFAULT])},
)
def test_service_ibmq_compile(mock_ibmq_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    assert service.ibmq_compile(cirq.Circuit()) == mock.DEFAULT
    assert service.ibmq_compile([cirq.Circuit()]) == [mock.DEFAULT]

    with mock.patch.dict("sys.modules", {"unittest": None}), pytest.raises(
        applications_superstaq.SuperstaQModuleNotFoundException,
        match="'ibmq_compile' requires module 'unittest'",
    ):
        _ = service.ibmq_compile(cirq.Circuit())


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.neutral_atom_compile",
    return_value={"pulses": applications_superstaq.converters.serialize([mock.DEFAULT])},
)
def test_service_neutral_atom_compile(mock_neutral_atom_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    print(service.neutral_atom_compile(cirq.Circuit()))
    assert service.neutral_atom_compile(cirq.Circuit()) == mock.DEFAULT
    assert service.neutral_atom_compile([cirq.Circuit()]) == [mock.DEFAULT]

    with mock.patch.dict("sys.modules", {"unittest": None}), pytest.raises(
        applications_superstaq.SuperstaQModuleNotFoundException,
        match="'neutral_atom_compile' requires module 'unittest'",
    ):
        _ = service.neutral_atom_compile(cirq.Circuit())


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
