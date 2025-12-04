# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from __future__ import annotations

import collections
import datetime
import json
import os
import textwrap
import uuid
from unittest import mock
from unittest.mock import patch

import cirq
import general_superstaq as gss
import numpy as np
import pandas as pd
import pytest
import qiskit
import qiskit_superstaq as qss
import sympy
from general_superstaq import ResourceEstimate
from general_superstaq.superstaq_client import _SuperstaqClient, _SuperstaqClientV3

import cirq_superstaq as css


@pytest.mark.parametrize("gate", [cirq.Y, cirq.CX, cirq.CCZ, css.QutritZ0, css.BSWAP])
def test_to_matrix_gate(gate: cirq.Gate) -> None:
    matrix = cirq.unitary(gate)
    qid_shape = cirq.qid_shape(gate)
    assert css.service._to_matrix_gate(matrix) == cirq.MatrixGate(matrix, qid_shape=qid_shape)


def test_to_matrix_gate_error() -> None:
    matrix = np.eye(5)
    with pytest.raises(ValueError, match=r"Could not determine qid_shape"):
        _ = css.service._to_matrix_gate(matrix)


def test_counts_to_results() -> None:
    qubits = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        cirq.H(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0]),
        cirq.measure(qubits[1]),
    )
    result = css.service.counts_to_results({"01": 1, "11": 2}, circuit, cirq.ParamResolver({}))
    assert result.histogram(key="q(0)q(1)") == collections.Counter({3: 2, 1: 1})

    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0], key="0"),
        cirq.measure(qubits[1], key="1"),
    )
    result = css.service.counts_to_results({"00": 50, "11": 50}, circuit, cirq.ParamResolver({}))
    assert result.histogram(key="01") == collections.Counter({0: 50, 3: 50})

    result = css.service.counts_to_results(
        {"00": 50.0, "11": 50.0}, circuit, cirq.ParamResolver({})
    )
    assert result.histogram(key="01") == collections.Counter({0: 50, 3: 50})

    with pytest.warns(UserWarning, match=r"raw counts contain fractional"):
        result = css.service.counts_to_results(
            {"00": 50.1, "11": 49.9}, circuit, cirq.ParamResolver({})
        )
    assert result.histogram(key="01") == collections.Counter({0: 50, 3: 50})

    with pytest.warns(UserWarning, match=r"raw counts contain negative"):
        result = css.service.counts_to_results(
            {"00": -50, "11": 100}, circuit, cirq.ParamResolver({})
        )
    assert result.histogram(key="01") == collections.Counter({3: 100})


def test_service_wrong_version() -> None:
    with pytest.raises(ValueError, match=r"`api_version` can only take value 'v0.2.0' or 'v0.3.0'"):
        css.Service(api_version="v4")


@pytest.mark.parametrize("api_version", ["v0.2.0", "v0.3.0"])
def test_service_resolve_target(api_version: str) -> None:
    service = css.Service(api_key="key", default_target="ss_bar_qpu", api_version=api_version)
    assert service._resolve_target("ss_foo_qpu") == "ss_foo_qpu"
    assert service._resolve_target(None) == "ss_bar_qpu"

    service = css.Service(api_key="key", api_version=api_version)
    assert service._resolve_target("ss_foo_qpu") == "ss_foo_qpu"
    with pytest.raises(ValueError, match=r"requires a target"):
        _ = service._resolve_target(None)


def test_service_run_and_get_counts() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    assert isinstance(service._client, _SuperstaqClient)
    mock_client = mock.MagicMock(spec=_SuperstaqClient)
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "Ready",
    }
    job_dict = {
        "data": {"histogram": {"11": 1}},
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
        "target": "ss_unconstrained_simulator",
    }
    mock_client.fetch_jobs.return_value = {"job_id": job_dict}

    service._client = mock_client

    a = sympy.Symbol("a")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X**a)(q), cirq.measure(q, key="a"))
    params = cirq.ParamResolver({"a": 0.5})
    counts = service.get_counts(
        circuits=circuit,
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert counts == {"11": 1}

    results = service.run(
        circuits=circuit,
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert results.histogram(key="a") == collections.Counter({3: 1})

    # Multiple circuit run
    mock_client.create_job.return_value = {
        "job_ids": ["job_id_1", "job_id_2"],
        "status": "Done",
        "data": {"histogram": {"11": 1}},
        "samples": {"11": 1},
    }
    mock_client.fetch_jobs.return_value = {"job_id_1": job_dict, "job_id_2": job_dict}

    service._client = mock_client
    multi_results = service.run(
        circuits=[circuit, circuit],
        repetitions=10,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )

    assert isinstance(multi_results, list)
    for result in multi_results:
        assert result.histogram(key="a") == collections.Counter({3: 1})

    multi_counts = service.get_counts(
        circuits=[circuit, circuit],
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert multi_counts == [{"11": 1}, {"11": 1}]


def test_service_run_and_get_countsV3() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com", api_version="v0.3.0")
    assert isinstance(service._client, _SuperstaqClientV3)
    mock_client = mock.MagicMock(spec=_SuperstaqClientV3)
    job_id = uuid.UUID(int=42)
    mock_client.create_job.return_value = {"job_id": job_id, "num_circuits": 1}
    job_dict = {
        "job_type": "simulate",
        "statuses": ["completed"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_unconstrained_simulator",
        "provider_id": ["provider_id"],
        "num_circuits": 1,
        "compiled_circuits": ["compiled circuit"],
        "input_circuits": ["input circuit"],
        "circuit_type": "cirq",
        "counts": [{"11": 1}],
        "results_dicts": [],
        "shots": [1],
        "dry_run": True,
        "submission_timestamp": datetime.datetime.now(),
        "last_updated_timestamp": [datetime.datetime.now()],
        "initial_logical_to_physicals": [{0: 0}],
        "final_logical_to_physicals": [{0: 0}],
        "logical_qubits": ["0"],
        "physical_qubits": ["0"],
    }
    mock_client.fetch_jobs.return_value = {str(job_id): job_dict}

    service._client = mock_client

    a = sympy.Symbol("a")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X**a)(q), cirq.measure(q, key="a"))
    params = cirq.ParamResolver({"a": 0.5})
    counts = service.get_counts(
        circuits=circuit,
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert counts == {"11": 1}

    results = service.run(
        circuits=circuit,
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert results.histogram(key="a") == collections.Counter({3: 1})

    # Multiple circuit run
    mock_client.create_job.return_value = {"job_id": job_id, "num_circuits": 2}
    job_dict = {
        "job_type": "simulate",
        "statuses": ["completed"] * 2,
        "status_messages": [None] * 2,
        "user_email": "test@email.com",
        "target": "ss_unconstrained_simulator",
        "provider_id": ["provider_id"] * 2,
        "num_circuits": 1,
        "compiled_circuits": ["compiled circuit"] * 2,
        "input_circuits": ["input circuit"] * 2,
        "circuit_type": "cirq",
        "counts": [{"11": 1}] * 2,
        "results_dicts": [],
        "shots": [1] * 2,
        "dry_run": True,
        "submission_timestamp": datetime.datetime.now(),
        "last_updated_timestamp": [datetime.datetime.now()] * 2,
        "initial_logical_to_physicals": [{0: 0}] * 2,
        "final_logical_to_physicals": [{0: 0}] * 2,
        "logical_qubits": ["0"] * 2,
        "physical_qubits": ["0"] * 2,
    }
    mock_client.fetch_jobs.return_value = {str(job_id): job_dict}

    service._client = mock_client
    multi_results = service.run(
        circuits=[circuit, circuit],
        repetitions=10,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )

    assert isinstance(multi_results, list)
    for result in multi_results:
        assert result.histogram(key="a") == collections.Counter({3: 1})

    multi_counts = service.get_counts(
        circuits=[circuit, circuit],
        repetitions=4,
        target="ss_unconstrained_simulator",
        param_resolver=params,
    )
    assert multi_counts == [{"11": 1}, {"11": 1}]


def test_service_sampler() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    assert isinstance(service._client, _SuperstaqClient)
    mock_client = mock.MagicMock(spec=_SuperstaqClient)
    service._client = mock_client
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "Ready",
    }
    job_dict = {
        "data": {"histogram": {"0": 3, "1": 1}},
        "num_qubits": 1,
        "samples": {"0": 3, "1": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "ss_unconstrained_simulator",
    }
    mock_client.fetch_jobs.return_value = {"job_id": job_dict}

    sampler = service.sampler(target="ss_unconstrained_simulator")
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="a"))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=["a"], index=[0, 1, 2, 3], data=[[0], [0], [0], [1]])
    )
    mock_client.create_job.assert_called_once()


def test_service_samplerV3() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com", api_version="v0.3.0")
    assert isinstance(service._client, _SuperstaqClientV3)
    mock_client = mock.MagicMock(spec=_SuperstaqClientV3)
    job_id = uuid.UUID(int=42)
    mock_client.create_job.return_value = {"job_id": job_id, "num_circuits": 1}

    job_dict = {
        "job_type": "simulate",
        "statuses": ["completed"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_unconstrained_simulator",
        "provider_id": ["provider_id"],
        "num_circuits": 1,
        "compiled_circuits": ["compiled circuit"],
        "input_circuits": ["input circuit"],
        "circuit_type": "cirq",
        "counts": [{"0": 3, "1": 1}],
        "results_dicts": [],
        "shots": [1],
        "dry_run": True,
        "submission_timestamp": datetime.datetime.now(),
        "last_updated_timestamp": [datetime.datetime.now()],
        "initial_logical_to_physicals": [{0: 0}],
        "final_logical_to_physicals": [{0: 0}],
        "logical_qubits": ["0"],
        "physical_qubits": ["0"],
    }
    mock_client.fetch_jobs.return_value = {str(job_id): job_dict}
    service._client = mock_client

    sampler = service.sampler(target="ss_unconstrained_simulator")
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="a"))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=["a"], index=[0, 1, 2, 3], data=[[0], [0], [0], [1]])
    )
    mock_client.create_job.assert_called_once()


def test_service_get_job() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    assert isinstance(service._client, _SuperstaqClient)
    mock_client = mock.MagicMock(spec=_SuperstaqClient)
    job_dict = {"status": "Ready"}
    mock_client.fetch_jobs.return_value = {"job_id": job_dict}
    service._client = mock_client

    job = service.get_job("job_id")

    # fetch_jobs() should not be called upon construction
    assert job.job_id() == "job_id"
    mock_client.fetch_jobs.assert_not_called()

    # ...but it will be called with the initial query of status()
    assert job.status() == "Ready"
    mock_client.fetch_jobs.assert_called_once_with(["job_id"])


def test_service_get_jobV3() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com", api_version="v0.3.0")
    assert isinstance(service._client, _SuperstaqClientV3)
    mock_client = mock.MagicMock(spec=_SuperstaqClientV3)
    job_dict = {
        "job_type": "simulate",
        "statuses": ["completed"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_example_qpu",
        "provider_id": ["provider_id"],
        "num_circuits": 1,
        "compiled_circuits": ["compiled circuit"],
        "input_circuits": ["input circuit"],
        "circuit_type": "cirq",
        "counts": [{"0": 1}],
        "results_dicts": [],
        "shots": [1],
        "dry_run": True,
        "submission_timestamp": datetime.datetime.now(),
        "last_updated_timestamp": [datetime.datetime.now()],
        "initial_logical_to_physicals": [{0: 0}],
        "final_logical_to_physicals": [{0: 0}],
        "logical_qubits": ["0"],
        "physical_qubits": ["0"],
    }
    job_id = uuid.UUID(int=0)
    mock_client.fetch_jobs.return_value = {str(job_id): job_dict}
    service._client = mock_client

    job = service.get_job(job_id)

    assert job.job_id() == job_id
    assert job.status() == gss.models.CircuitStatus.COMPLETED
    mock_client.fetch_jobs.assert_called_once_with([job_id])


def test_service_create_job() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    assert isinstance(service._client, _SuperstaqClient)
    mock_client = mock.MagicMock(spec=_SuperstaqClient)
    mock_client.create_job.return_value = {"job_ids": ["job_id"], "status": "Ready"}
    mock_client.fetch_jobs.return_value = {"job_id": {"status": "Done"}}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0)))
    job = service.create_job(
        circuits=circuit,
        repetitions=100,
        target="ss_fake_qpu",
        method="fake_method",
        fake_data="",
    )
    assert job.status() == "Done"
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "ss_fake_qpu"
    assert create_job_kwargs["method"] == "fake_method"
    assert create_job_kwargs["fake_data"] == ""


def test_service_create_jobV3() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com", api_version="v0.3.0")
    assert isinstance(service._client, _SuperstaqClientV3)
    mock_client = mock.MagicMock(spec=_SuperstaqClientV3)
    job_id = uuid.UUID(int=42)
    mock_client.create_job.return_value = {"job_id": job_id, "num_circuits": 1}
    job_dict = {
        "job_type": "simulate",
        "statuses": ["completed"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_example_qpu",
        "provider_id": ["provider_id"],
        "num_circuits": 1,
        "compiled_circuits": ["compiled circuit"],
        "input_circuits": ["input circuit"],
        "circuit_type": "cirq",
        "counts": [{"0": 1}],
        "results_dicts": [],
        "shots": [1],
        "dry_run": True,
        "submission_timestamp": datetime.datetime.now(),
        "last_updated_timestamp": [datetime.datetime.now()],
        "initial_logical_to_physicals": [{0: 0}],
        "final_logical_to_physicals": [{0: 0}],
        "logical_qubits": ["0"],
        "physical_qubits": ["0"],
    }
    mock_client.fetch_jobs.return_value = {str(job_id): job_dict}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0)))
    job = service.create_job(
        circuits=circuit,
        repetitions=100,
        target="ss_fake_qpu",
        method="fake_method",
        verbatim=True,
        fake_data="",
    )
    assert job.status() == gss.models.CircuitStatus.COMPLETED
    create_job_kwargs = mock_client.create_job.call_args[1]
    # # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "ss_fake_qpu"
    assert create_job_kwargs["method"] == "fake_method"
    assert create_job_kwargs["fake_data"] == ""
    assert create_job_kwargs["verbatim"] is True


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits(cirq.Circuit()),
        "initial_logical_to_physicals": cirq.to_json([[]]),
        "final_logical_to_physicals": cirq.to_json([[]]),
    },
)
def test_service_aqt_compile_single(mock_post_request: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.aqt_compile(cirq.Circuit(), test_options="yes")
    mock_post_request.assert_called_once_with(
        "/aqt_compile",
        {
            "cirq_circuits": css.serialization.serialize_circuits(cirq.Circuit()),
            "target": "aqt_keysight_qpu",
            "options": '{\n  "test_options": "yes"\n}',
        },
    )

    alt_out = service.compile(cirq.Circuit(), target="aqt_keysight_qpu", test_options="yes")

    for output in [out, alt_out]:
        assert output.circuit == cirq.Circuit()
        assert output.initial_logical_to_physical == {}
        assert output.final_logical_to_physical == {}
        assert not hasattr(output, "circuits")
        assert not hasattr(output, "initial_logical_to_physicals")
        assert not hasattr(output, "final_logical_to_physicals")

    with pytest.raises(ValueError, match=r"Unable to serialize configuration"):
        _ = service.aqt_compile(cirq.Circuit(), atol=1e-2, pulses=123, variables=456)

    gate_defs = {
        "CZ3": css.CZ3,
        "CZ3/T5C4": None,
        "CS/simul": css.ParallelGates(cirq.CZ, cirq.CZ).on(*cirq.LineQubit.range(4, 8)),
        "CS2": cirq.unitary(cirq.CZ**0.49),
        "CS3": cirq.unitary(css.CZ3**0.5),
    }
    out = service.aqt_compile(
        cirq.Circuit(),
        gate_defs=gate_defs,
        gateset={"CZ3": [[5, 6]], "X90": [[5], [6]], "EFX90": [[5], [6]]},
        atol=1e-3,
        aqt_configs={},
    )
    expected_options = {
        "aqt_configs": {},
        "atol": 1e-3,
        "gate_defs": {
            "CZ3": css.CZ3,
            "CZ3/T5C4": None,
            "CS/simul": css.ParallelGates(cirq.CZ, cirq.CZ).on(*cirq.LineQubit.range(4, 8)),
            "CS2": cirq.MatrixGate(cirq.unitary(cirq.CZ**0.49)),
            "CS3": cirq.MatrixGate(cirq.unitary(css.CZ3**0.5), qid_shape=(3, 3)),
        },
        "gateset": {"CZ3": [[5, 6]], "X90": [[5], [6]], "EFX90": [[5], [6]]},
    }
    mock_post_request.assert_called_with(
        "/aqt_compile",
        {
            "cirq_circuits": css.serialization.serialize_circuits(cirq.Circuit()),
            "target": "aqt_keysight_qpu",
            "options": cirq.to_json(expected_options),
        },
    )
    assert out.circuit == cirq.Circuit()
    assert not hasattr(out, "circuits")

    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid AQT target."):
        service.aqt_compile(cirq.Circuit(), target="ss_example_qpu")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits([cirq.Circuit(), cirq.Circuit()]),
        "initial_logical_to_physicals": cirq.to_json([[], []]),
        "final_logical_to_physicals": cirq.to_json([[], []]),
    },
)
def test_service_aqt_compile_multiple(mock_post_request: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.aqt_compile([cirq.Circuit(), cirq.Circuit()], atol=1e-2)
    mock_post_request.assert_called_once()
    assert out.circuits == [cirq.Circuit(), cirq.Circuit()]
    assert out.initial_logical_to_physicals == [{}, {}]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits([cirq.Circuit()]),
        "initial_logical_to_physicals": cirq.to_json([[]]),
        "final_logical_to_physicals": cirq.to_json([[]]),
    },
)
def test_service_aqt_compile_eca(mock_post_request: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.aqt_compile(cirq.Circuit(), num_eca_circuits=1, random_seed=1234, atol=1e-2)
    mock_post_request.assert_called_once()
    assert out.circuits == [cirq.Circuit()]
    assert out.initial_logical_to_physicals == [{}]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    out = service.aqt_compile([cirq.Circuit()], num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [[cirq.Circuit()]]
    assert out.initial_logical_to_physicals == [[{}]]
    assert out.final_logical_to_physicals == [[{}]]

    with pytest.warns(DeprecationWarning, match=r"has been deprecated"):
        deprecated_out = service.aqt_compile_eca(
            [cirq.Circuit()], num_equivalent_circuits=1, random_seed=1234, atol=1e-2
        )
    assert deprecated_out.circuits == out.circuits
    assert deprecated_out.initial_logical_to_physicals == out.initial_logical_to_physicals
    assert deprecated_out.final_logical_to_physicals == out.final_logical_to_physicals


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_service_resource_estimate(mock_resource_estimate: mock.MagicMock) -> None:
    service = css.Service(remote_host="http://example.com", api_key="key")

    resource_estimate = ResourceEstimate(0, 1, 2)

    mock_resource_estimate.return_value = {
        "resource_estimates": [{"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2}]
    }

    assert (
        service.resource_estimate(cirq.Circuit(), "ss_unconstrained_simulator") == resource_estimate
    )


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_service_resource_estimate_list(mock_resource_estimate: mock.MagicMock) -> None:
    service = css.Service(remote_host="http://example.com", api_key="key")

    resource_estimates = [ResourceEstimate(0, 1, 2), ResourceEstimate(3, 4, 5)]

    mock_resource_estimate.return_value = {
        "resource_estimates": [
            {"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2},
            {"num_single_qubit_gates": 3, "num_two_qubit_gates": 4, "depth": 5},
        ]
    }

    assert (
        service.resource_estimate([cirq.Circuit()], "ss_unconstrained_simulator")
        == resource_estimates
    )


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient.qscout_compile")
def test_service_qscout_compile_single(mock_qscout_compile: mock.MagicMock) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {q0: q0}
    final_logical_to_physical = {q0: q0}

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
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, test_options="yes")
    alt_out = service.compile(circuit, target="qscout_peregrine_qpu", test_options="yes")
    assert out.circuit == circuit
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.jaqal_program == jaqal_program

    assert alt_out.circuit == circuit
    assert alt_out.initial_logical_to_physical == initial_logical_to_physical
    assert alt_out.final_logical_to_physical == final_logical_to_physical
    assert alt_out.jaqal_program == jaqal_program

    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid QSCOUT target."):
        service.qscout_compile(cirq.Circuit(), target="ss_example_qpu")


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient.qscout_compile")
def test_service_qscout_compile_multiple(mock_qscout_compile: mock.MagicMock) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        cirq.Circuit(cirq.H(q0), cirq.measure(q0)),
        cirq.Circuit(cirq.ISWAP(q0, q1)),
    ]
    initial_logical_to_physicals = [{q0: q0}, {q0: q0, q1: q1}]
    final_logical_to_physicals = [{q0: q0}, {q0: q1, q1: q0}]

    jaqal_programs = ["jaqal", "programs"]

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuits),
        "initial_logical_to_physicals": cirq.to_json(
            [list(l2p.items()) for l2p in initial_logical_to_physicals]
        ),
        "final_logical_to_physicals": cirq.to_json(
            [list(l2p.items()) for l2p in final_logical_to_physicals]
        ),
        "jaqal_programs": jaqal_programs,
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuits)
    assert out.circuits == circuits
    assert out.initial_logical_to_physicals == initial_logical_to_physicals
    assert out.final_logical_to_physicals == final_logical_to_physicals
    assert out.jaqal_programs == jaqal_programs

    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": "xx",
        "num_qubits": 2,
    }

    with pytest.raises(ValueError, match=r"At least 2 qubits are required"):
        _ = service.qscout_compile(circuits, num_qubits=1)


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient.qscout_compile")
@pytest.mark.parametrize("mirror_swaps", [True, False])
def test_qscout_compile_swap_mirror(
    mock_qscout_compile: mock.MagicMock, mirror_swaps: bool
) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, mirror_swaps=mirror_swaps)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == final_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_qscout_compile.assert_called_once()
    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": "xx",
        "num_qubits": 1,
    }


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient.qscout_compile")
def test_qscout_compile_error_rates(mock_qscout_compile: mock.MagicMock) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, error_rates={(0, 1): 0.3, (0, 2): 0.2, (1,): 0.1})
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == final_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_qscout_compile.assert_called_once()
    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "base_entangling_gate": "xx",
        "mirror_swaps": False,
        "error_rates": [[[0, 1], 0.3], [[0, 2], 0.2], [[1], 0.1]],
        "num_qubits": 3,
    }


@mock.patch("general_superstaq.superstaq_client._SuperstaqClient.qscout_compile")
@pytest.mark.parametrize("base_entangling_gate", ["xx", "zz"])
def test_qscout_compile_base_entangling_gate(
    mock_qscout_compile: mock.MagicMock, base_entangling_gate: str
) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, base_entangling_gate=base_entangling_gate)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == final_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_qscout_compile.assert_called_once()
    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": base_entangling_gate,
        "num_qubits": 1,
    }


def test_qscout_compile_wrong_base_entangling_gate() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))

    service = css.Service(api_key="key", remote_host="http://example.com")
    with pytest.raises(ValueError, match=r"`base_entangling_gate` must be"):
        _ = service.qscout_compile(circuit, base_entangling_gate="yy")


@mock.patch("requests.Session.post")
def test_qscout_compile_num_qubits(mock_post: mock.MagicMock) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_post.return_value.json = lambda: {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, num_qubits=5)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == final_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_post.assert_called_once()
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": "xx",
        "num_qubits": 5,
    }


@mock.patch("requests.Session.post")
def test_service_cq_compile_single(mock_post: mock.MagicMock) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {cirq.q(0): cirq.q(0)}
    final_logical_to_physical = {cirq.q(10): cirq.q(0)}

    mock_post.return_value.json = lambda: {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.cq_compile(circuit, test_options="yes")
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical

    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid CQ target."):
        service.cq_compile(cirq.Circuit(), target="ss_example_qpu")


@mock.patch("requests.Session.post")
def test_service_ibmq_compile(mock_post: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    initial_logical_to_physical = {cirq.q(0): cirq.q(0)}
    final_logical_to_physical = {cirq.q(4): cirq.q(0)}

    mock_post.return_value.json = lambda: {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qiskit.QuantumCircuit()]),
        "initial_logical_to_physicals": cirq.to_json([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    assert (
        service.ibmq_compile(
            circuit, dd_strategy="standard", test_options="yes", target="ibmq_fake_qpu"
        ).circuit
        == circuit
    )

    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "dd_strategy": "standard",
        "dynamical_decoupling": True,
        "test_options": "yes",
    }

    assert service.ibmq_compile([circuit], target="ibmq_fake_qpu").circuits == [circuit]
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "dd_strategy": "adaptive",
        "dynamical_decoupling": True,
    }
    assert (
        service.ibmq_compile(circuit, target="ibmq_fake_qpu").pulse_gate_circuit
        == qiskit.QuantumCircuit()
    )
    assert service.ibmq_compile([circuit], target="ibmq_fake_qpu").pulse_gate_circuits == [
        qiskit.QuantumCircuit()
    ]
    assert (
        service.ibmq_compile(circuit, target="ibmq_fake_qpu").initial_logical_to_physical
        == initial_logical_to_physical
    )
    assert service.ibmq_compile([circuit], target="ibmq_fake_qpu").initial_logical_to_physicals == [
        initial_logical_to_physical
    ]
    assert (
        service.ibmq_compile(circuit, target="ibmq_fake_qpu").final_logical_to_physical
        == final_logical_to_physical
    )
    assert service.ibmq_compile([circuit], target="ibmq_fake_qpu").final_logical_to_physicals == [
        final_logical_to_physical
    ]

    with mock.patch.dict("sys.modules", {"qiskit_superstaq": None}):
        with pytest.warns(UserWarning, match=r"qiskit-superstaq is required"):
            assert (
                service.ibmq_compile(cirq.Circuit(), target="ibmq_fake_qpu").pulse_gate_circuit
                is None
            )
        with pytest.warns(UserWarning, match=r"qiskit-superstaq is required"):
            assert (
                service.ibmq_compile([cirq.Circuit()], target="ibmq_fake_qpu").pulse_gate_circuits
                is None
            )

    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid IBMQ target."):
        service.ibmq_compile(cirq.Circuit(), target="ss_example_qpu")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.supercheq",
)
def test_service_supercheq(mock_supercheq: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    circuits = [cirq.Circuit()]
    fidelities = np.array([1])
    mock_supercheq.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuits),
        "fidelities": gss.serialization.serialize(fidelities),
    }
    assert service.supercheq([[0]], 1, 1) == (circuits, fidelities)


@mock.patch("requests.Session.post")
def test_service_dfe(mock_post: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    circuit = cirq.Circuit(cirq.X(cirq.q(0)))
    mock_post.return_value.json = lambda: ["id1", "id2"]
    assert service.submit_dfe(
        rho_1=(circuit, "ss_example_qpu"),
        rho_2=(circuit, "ss_example_qpu"),
        num_random_bases=5,
        shots=100,
    ) == ["id1", "id2"]

    with pytest.raises(TypeError, match=r"should contain a single `cirq.Circuit`"):
        service.submit_dfe(
            rho_1=([circuit, circuit], "ss_example_qpu"),  # type: ignore[arg-type]
            rho_2=(circuit, "ss_example_qpu"),
            num_random_bases=5,
            shots=100,
        )

    mock_post.return_value.json = lambda: 1
    assert service.process_dfe(["1", "2"]) == 1


@mock.patch("requests.Session.post")
def test_aces(
    mock_post: mock.MagicMock,
) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_post.return_value.json = lambda: "id1"
    assert (
        service.submit_aces(
            target="ss_unconstrained_simulator",
            qubits=[0, 1],
            shots=100,
            num_circuits=10,
            mirror_depth=5,
            extra_depth=5,
            noise=cirq.NoiseModel.from_noise_model_like(cirq.depolarize(0.1)),
        )
        == "id1"
    )

    assert (
        service.submit_aces(
            target="ss_unconstrained_simulator",
            qubits=[0, 1],
            shots=100,
            num_circuits=10,
            mirror_depth=5,
            extra_depth=5,
            weights=[1, 2],
            noise="asymmetric_depolarize",
            error_prob=(0.1, 0.1, 0.1),
        )
        == "id1"
    )

    mock_post.return_value.json = lambda: [1] * 51
    assert service.process_aces("id1") == [1] * 51


@mock.patch("requests.Session.post")
def test_cb(
    mock_post: mock.MagicMock,
) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_post.return_value.json = lambda: "id1"
    assert (
        service.submit_cb(
            target="ss_unconstrained_simulator",
            repetitions=50,
            process_circuit=cirq.Circuit(),
            n_channels=6,
            n_sequences=30,
            depths=[1, 2, 3],
            method="dry-run",
            noise=cirq.NoiseModel.from_noise_model_like(cirq.depolarize(0.1)),
            error_prob=(0.1, 0.1, 0.1),
        )
        == "id1"
    )

    assert (
        service.submit_cb(
            target="ss_unconstrained_simulator",
            repetitions=50,
            process_circuit=cirq.Circuit(),
            n_channels=6,
            n_sequences=30,
            depths=[1, 2, 3],
            method="dry-run",
            noise="asymmetric_depolarize",
        )
        == "id1"
    )

    test_data = {
        "circuit_data": {
            "ps": {
                "depth_1": {
                    "seq": {
                        "result": "{}",
                        "c_of_p": "{}",
                        "circuit": "{}",
                        "compiled_circuit": "{}",
                    }
                }
            }
        },
        "instance_information": cirq.to_json(
            {
                "target": "ss_unconstrained_simulator",
                "depths": [1, 2, 3],
                "n_channels": 2,
            }
        ),
        "process_fidelity_data": {
            "averages": {
                "test1": [1.0, 1.0, 1.0],
                "test2": [1.0, 1.0, 1.0],
                "test3": [1.0, 1.0, 1.0],
            },
            "std_devs": {
                "test1": {"depth=1": 0.0, "depth=2": 0.0, "depth=3": 0.0},
                "test2": {"depth=1": 0.0, "depth=2": 0.0, "depth=3": 0.0},
                "test3": {"depth=1": 0.0, "depth=2": 0.0, "depth=3": 0.0},
            },
            "evs": {
                "test1": {"depth=1": [1.0], "depth=2": [1.0], "depth=3": [1.0]},
                "test2": {"depth=1": [1.0], "depth=2": [1.0], "depth=3": [1.0]},
                "test3": {"depth=1": [1.0], "depth=2": [1.0], "depth=3": [1.0]},
            },
        },
    }
    processed_test_data = {
        "circuit_data": {
            "ps": {
                "depth_1": {
                    "seq": {"result": {}, "c_of_p": {}, "circuit": {}, "compiled_circuit": "{}"}
                }
            }
        },
        "instance_information": {
            "target": "ss_unconstrained_simulator",
            "depths": [1, 2, 3],
            "n_channels": 2,
        },
        "process_fidelity_data": test_data["process_fidelity_data"],
        "fit_data": {
            "A_test1": 1.0,
            "p_test1": 1.0,
            "A_test2": 1.0,
            "p_test2": 1.0,
            "A_test3": 1.0,
            "p_test3": 1.0,
            "e_f": -0.5,
        },
    }

    mock_post.return_value.json = lambda: test_data
    assert service.process_cb("id1") == processed_test_data

    with patch(
        "matplotlib.pyplot.show",
        return_value={"test": 123},
    ):
        service.plot(processed_test_data)

    # Test truncated labels
    processed_test_data["instance_information"] = {"depths": [1, 2], "n_channels": 11}

    with patch(
        "matplotlib.pyplot.show",
        return_value={"test": 123},
    ):
        service.plot(processed_test_data)


@mock.patch("requests.Session.post")
def test_service_target_info(mock_post: mock.MagicMock) -> None:
    fake_data = {"target_info": {"backend_name": "ss_example_qpu", "max_experiments": 1234}}
    mock_post.return_value.json = lambda: fake_data
    service = css.Service(api_key="key", remote_host="http://example.com")
    assert service.target_info("ss_example_qpu") == fake_data["target_info"]


@mock.patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "tomyheart"})
def test_service_api_key_via_env() -> None:
    service = css.Service(remote_host="http://example.com")
    assert service._client.api_key == "tomyheart"


@mock.patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "http://example.com"})
def test_service_remote_host_via_env() -> None:
    service = css.Service("tomyheart")
    assert service._client.remote_host == "http://example.com"


@mock.patch.dict(os.environ, clear=True)
def test_service_no_url_default() -> None:
    service = css.Service("tomyheart")
    assert service._client.remote_host == gss.API_URL
