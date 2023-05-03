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
# pylint: disable=missing-function-docstring

import collections
import json
import os
import re
import textwrap
from unittest import mock

import cirq
import general_superstaq as gss
import numpy as np
import pandas as pd
import pytest
import sympy
from general_superstaq import ResourceEstimate

import cirq_superstaq as css


@pytest.mark.parametrize("gate", [cirq.Y, cirq.CX, cirq.CCZ, css.QutritZ0, css.BSWAP])
def test_to_matrix_gate(gate: cirq.Gate) -> None:
    matrix = cirq.unitary(gate)
    qid_shape = cirq.qid_shape(gate)
    assert css.service._to_matrix_gate(matrix) == cirq.MatrixGate(matrix, qid_shape=qid_shape)


def test_to_matrix_gate_error() -> None:
    matrix = np.eye(5)
    with pytest.raises(ValueError, match="Could not determine qid_shape"):
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


def test_validate_cirq_circuits() -> None:
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.service._validate_cirq_circuits("circuit_invalid")

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.service._validate_cirq_circuits([circuit, "circuit_invalid"])


def test_validate_integer_param() -> None:

    # Tests for valid inputs -> Pass
    valid_inputs = [1, 10, "10", 10.0, np.int16(10), 0b1010]
    for input_value in valid_inputs:
        css.service._validate_integer_param(input_value)

    # Tests for invalid input -> TypeError
    invalid_inputs = [None, "reps", "{!r}".format(b"invalid"), 1.5, "1.0", {1}, [1, 2, 3], "0b1010"]
    for input_value in invalid_inputs:
        with pytest.raises(TypeError) as msg:
            css.service._validate_integer_param(input_value)
        assert re.search(
            re.escape(f"{input_value} cannot be safely cast as an integer."), str(msg.value)
        )

    # Tests for invalid input -> ValueError
    invalid_values = [0, -1]
    for input_value in invalid_values:
        with pytest.raises(
            ValueError,
            match="Must be a positive integer.",
        ):
            css.service._validate_integer_param(input_value)


def test_service_resolve_target() -> None:
    service = css.Service(api_key="key", default_target="ss_bar_qpu")
    assert service._resolve_target("ss_foo_qpu") == "ss_foo_qpu"
    assert service._resolve_target(None) == "ss_bar_qpu"

    service = css.Service(api_key="key")
    assert service._resolve_target("ss_foo_qpu") == "ss_foo_qpu"
    with pytest.raises(ValueError, match="requires a target"):
        _ = service._resolve_target(None)


def test_service_run_and_get_counts() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }
    mock_client.get_job.return_value = {
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

    service._client = mock_client

    a = sympy.Symbol("a")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X**a)(q), cirq.measure(q, key="a"))
    params = cirq.ParamResolver({"a": 0.5})
    counts = service.get_counts(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        param_resolver=params,
    )
    assert counts == {"11": 1}

    result = service.run(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        param_resolver=params,
    )
    assert result.histogram(key="a") == collections.Counter({3: 1})


def test_service_sampler() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    service._client = mock_client
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }
    mock_client.get_job.return_value = {
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
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    job_dict = {"status": "ready"}
    mock_client.get_job.return_value = job_dict
    service._client = mock_client

    job = service.get_job("job_id")

    # get_job() should not be called upon construction
    assert job.job_id() == "job_id"
    mock_client.get_job.assert_not_called()

    # ...but it will be called with the initial query of status()
    assert job.status() == "ready"
    mock_client.get_job.assert_called_once_with("job_id")


def test_service_create_job() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {"job_ids": ["job_id"], "status": "ready"}
    mock_client.get_job.return_value = {"status": "completed"}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0)))
    job = service.create_job(
        circuit=circuit,
        repetitions=100,
        target="ss_fake_qpu",
        method="fake_method",
        options={"fake_data": ""},
    )
    assert job.status() == "completed"
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "ss_fake_qpu"
    assert create_job_kwargs["method"] == "fake_method"
    assert create_job_kwargs["options"] == {"fake_data": ""}

    with pytest.raises(ValueError, match="Circuit has no measurements to sample"):
        service.create_job(cirq.Circuit())


def test_service_get_balance() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    service._client = mock_client

    assert service.get_balance() == "$12,345.68"
    assert service.get_balance(pretty_output=False) == 12345.6789


def test_service_get_targets() -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    targets = {
        "superstaq_targets": {
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
    mock_client.get_targets.return_value = targets
    service._client = mock_client

    assert service.get_targets() == targets["superstaq_targets"]


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits(cirq.Circuit()),
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
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
    assert out.circuit == cirq.Circuit()
    assert out.final_logical_to_physical == {}
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")
    assert not hasattr(out, "final_logical_to_physicals")

    gate_defs = {
        "CZ3": css.CZ3,
        "CZ3/T5C4": None,
        "CS/simul": css.ParallelGates(cirq.CZ, cirq.CZ).on(*cirq.LineQubit.range(4, 8)),
        "CS2": cirq.unitary(cirq.CZ**0.49),
        "CS3": cirq.unitary(css.CZ3**0.5),
    }
    out = service.aqt_compile(cirq.Circuit(), gate_defs=gate_defs, atol=1e-3)

    expected_options = {
        "atol": 1e-3,
        "gate_defs": {
            "CZ3": css.CZ3,
            "CZ3/T5C4": None,
            "CS/simul": css.ParallelGates(cirq.CZ, cirq.CZ).on(*cirq.LineQubit.range(4, 8)),
            "CS2": cirq.MatrixGate(cirq.unitary(cirq.CZ**0.49), name="CS2"),
            "CS3": cirq.MatrixGate(cirq.unitary(css.CZ3**0.5), qid_shape=(3, 3), name="CS3"),
        },
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
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits([cirq.Circuit(), cirq.Circuit()]),
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]], [[]]]),
        "final_logical_to_physicals": cirq.to_json([[], []]),
    },
)
def test_service_aqt_compile_multiple(mock_post_request: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.aqt_compile([cirq.Circuit(), cirq.Circuit()], atol=1e-2)
    mock_post_request.assert_called_once()
    assert out.circuits == [cirq.Circuit(), cirq.Circuit()]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
    assert not hasattr(out, "final_logical_to_physical")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.post_request",
    return_value={
        "cirq_circuits": css.serialization.serialize_circuits([cirq.Circuit()]),
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
        "final_logical_to_physicals": cirq.to_json([[]]),
    },
)
def test_service_aqt_compile_eca(mock_post_request: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.aqt_compile_eca(
        cirq.Circuit(), num_equivalent_circuits=1, random_seed=1234, atol=1e-2
    )
    mock_post_request.assert_called_once()
    assert out.circuits == [cirq.Circuit()]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
    assert not hasattr(out, "final_logical_to_physical")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.resource_estimate",
)
def test_service_resource_estimate(mock_resource_estimate: mock.MagicMock) -> None:
    service = css.Service(remote_host="http://example.com", api_key="key")

    resource_estimate = ResourceEstimate(0, 1, 2)

    mock_resource_estimate.return_value = {
        "resource_estimates": [{"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2}]
    }

    assert service.resource_estimate(cirq.Circuit(), "qasm_simulator") == resource_estimate


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.resource_estimate",
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

    assert service.resource_estimate([cirq.Circuit()], "qasm_simulator") == resource_estimates


@mock.patch("general_superstaq.superstaq_client._SuperstaQClient.qscout_compile")
def test_service_qscout_compile_single(mock_qscout_compile: mock.MagicMock) -> None:

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
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
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, test_options="yes")
    assert out.circuit == circuit
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program


@mock.patch("general_superstaq.superstaq_client._SuperstaQClient.qscout_compile")
@pytest.mark.parametrize("mirror_swaps", (True, False))
def test_qscout_compile_swap_mirror(
    mock_qscout_compile: mock.MagicMock, mirror_swaps: bool
) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, mirror_swaps=mirror_swaps)
    assert out.circuit == circuit
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_qscout_compile.assert_called_once()
    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": "xx",
    }


@mock.patch("general_superstaq.superstaq_client._SuperstaQClient.qscout_compile")
@pytest.mark.parametrize("base_entangling_gate", ("xx", "zz"))
def test_qscout_compile_base_entangling_gate(
    mock_qscout_compile: mock.MagicMock, base_entangling_gate: str
) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))
    final_logical_to_physical = {q0: q0}

    jaqal_program = ""

    mock_qscout_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.qscout_compile(circuit, base_entangling_gate=base_entangling_gate)
    assert out.circuit == circuit
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    mock_qscout_compile.assert_called_once()
    assert json.loads(mock_qscout_compile.call_args[0][0]["options"]) == {
        "mirror_swaps": True,
        "base_entangling_gate": base_entangling_gate,
    }


def test_qscout_compile_wrong_base_entangling_gate() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0))

    service = css.Service(api_key="key", remote_host="http://example.com")
    with pytest.raises(ValueError):
        _ = service.qscout_compile(circuit, base_entangling_gate="yy")


@mock.patch("general_superstaq.superstaq_client._SuperstaQClient.cq_compile")
def test_service_cq_compile_single(mock_cq_compile: mock.MagicMock) -> None:

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    final_logical_to_physical = {cirq.q(10): cirq.q(0)}

    mock_cq_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
        "options": {"test_options": "yes"},
    }

    service = css.Service(api_key="key", remote_host="http://example.com")
    out = service.cq_compile(circuit, test_options="yes")
    assert out.circuit == circuit
    assert out.final_logical_to_physical == final_logical_to_physical


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.ibmq_compile",
)
def test_service_ibmq_compile(mock_ibmq_compile: mock.MagicMock) -> None:
    service = css.Service(api_key="key", remote_host="http://example.com")

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    final_logical_to_physical = {cirq.q(4): cirq.q(0)}

    mock_ibmq_compile.return_value = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "pulses": gss.serialization.serialize([mock.DEFAULT]),
        "final_logical_to_physicals": cirq.to_json([list(final_logical_to_physical.items())]),
    }

    assert service.ibmq_compile(circuit, test_options="yes").circuit == circuit
    assert service.ibmq_compile([circuit]).circuits == [circuit]
    assert service.ibmq_compile(circuit).pulse_sequence == mock.DEFAULT
    assert service.ibmq_compile([circuit]).pulse_sequences == [mock.DEFAULT]
    assert service.ibmq_compile(circuit).final_logical_to_physical == final_logical_to_physical
    assert service.ibmq_compile([circuit]).final_logical_to_physicals == [final_logical_to_physical]

    with mock.patch.dict("sys.modules", {"qiskit": None}):
        assert service.ibmq_compile(cirq.Circuit()).pulse_sequence is None
        assert service.ibmq_compile([cirq.Circuit()]).pulse_sequences is None

    with pytest.raises(ValueError, match="not an IBMQ target"):
        _ = service.ibmq_compile(cirq.Circuit(), target="aqt_keysight_qpu")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.supercheq",
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


@mock.patch.dict(os.environ, {"SUPERSTAQ_API_KEY": "tomyheart"})
def test_service_api_key_via_env() -> None:
    service = css.Service(remote_host="http://example.com")
    assert service._client.api_key == "tomyheart"


@mock.patch.dict(os.environ, {"SUPERSTAQ_REMOTE_HOST": "http://example.com"})
def test_service_remote_host_via_env() -> None:
    service = css.Service("tomyheart")
    assert service._client.remote_host == "http://example.com"


@mock.patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_service_no_param_or_env_variable() -> None:
    with pytest.raises(EnvironmentError):
        _ = css.Service(remote_host="http://example.com")


@mock.patch.dict(os.environ, clear=True)
def test_service_no_url_default() -> None:
    service = css.Service("tomyheart")
    assert service._client.remote_host == gss.API_URL
