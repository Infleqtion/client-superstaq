# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import DEFAULT, MagicMock, patch

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss

if TYPE_CHECKING:
    from qiskit_superstaq.conftest import MockSuperstaqProvider


def test_run(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])

    backend = fake_superstaq_provider.get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=qc, shots=1000)
        expected = qss.SuperstaqJob(backend, "job_id")
        assert answer == expected

    with pytest.raises(ValueError, match="Circuit has no measurements to sample"):
        qc.remove_final_measurements()
        backend.run(qc, shots=1000)


def test_multi_circuit_run(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc1 = qiskit.QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = qiskit.QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])

    backend = fake_superstaq_provider.get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=[qc1, qc2], shots=1000)
        expected = qss.SuperstaqJob(backend, "job_id")
        assert answer == expected


def test_multi_arg_run(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])

    backend = fake_superstaq_provider.get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=qc, shots=1000, method="fake_method", test="123")
        expected = qss.SuperstaqJob(backend, "job_id")
        assert answer == expected


def test_retrieve_job(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])
    backend = fake_superstaq_provider.get_backend("ibmq_qasm_simulator")
    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        job = backend.run(qc, shots=1000)
    assert job == backend.retrieve_job("job_id")


def test_eq(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    backend1 = fake_superstaq_provider.get_backend("ibmq_qasm_simulator")
    assert backend1 != 3

    backend2 = fake_superstaq_provider.get_backend("ibmq_athens_qpu")
    assert backend1 != backend2

    backend3 = fake_superstaq_provider.get_backend("ibmq_qasm_simulator")
    assert backend1 == backend3


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock) -> None:
    # AQT compile
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    backend = provider.get_backend("aqt_keysight_qpu")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[1, 4]]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }
    out = backend.compile(qc)
    assert out.circuit == qc
    assert out.final_logical_to_physical == {1: 4}
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")
    mock_post.assert_called_with(
        f"{provider._client.url}/aqt_compile",
        headers=provider._client.headers,
        verify=provider._client.verify_https,
        json={
            "qiskit_circuits": qss.serialize_circuits(qc),
            "target": "aqt_keysight_qpu",
            "options": "{}",
        },
    )

    out = backend.compile([qc], atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{1: 4}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
    mock_post.assert_called_with(
        f"{provider._client.url}/aqt_compile",
        headers=provider._client.headers,
        verify=provider._client.verify_https,
        json={
            "qiskit_circuits": qss.serialize_circuits(qc),
            "target": "aqt_keysight_qpu",
            "options": json.dumps({"atol": 1e-2}),
        },
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]], [[]]]),
    }
    matrix = qiskit.circuit.library.CRXGate(1.23).to_matrix()
    out = backend.compile([qc, qc], gate_defs={"CRX": matrix})
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
    mock_post.assert_called_with(
        f"{provider._client.url}/aqt_compile",
        headers=provider._client.headers,
        verify=provider._client.verify_https,
        json={
            "qiskit_circuits": qss.serialize_circuits([qc, qc]),
            "target": "aqt_keysight_qpu",
            "options": qss.serialization.to_json({"gate_defs": {"CRX": matrix}}),
        },
    )

    with pytest.raises(ValueError, match="'aqt_keysight_qpu' is not a valid IBMQ target."):
        backend.ibmq_compile([qc])

    with pytest.raises(ValueError, match="'aqt_keysight_qpu' is not a valid Sandia target."):
        backend.qscout_compile([qc])

    with pytest.raises(ValueError, match="'aqt_keysight_qpu' is not a valid CQ target."):
        backend.cq_compile([qc])

    # AQT ECA compile
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }

    out = backend.compile(qc, num_eca_circuits=1, random_seed=1234, atol=1e-2, test_options="yes")
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    # AQT ECA compile
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }

    out = backend.compile(qc, num_eca_circuits=1, random_seed=1234, atol=1e-2, test_options="yes")
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


@patch("requests.post")
def test_ibmq_compile(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    backend = provider.get_backend("ibmq_jakarta_qpu")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    final_logical_to_physical = {0: 4, 1: 5}
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[0, 4], [1, 5]]]",
        "pulses": gss.serialization.serialize([DEFAULT]),
    }
    assert backend.compile(
        qiskit.QuantumCircuit(), test_options="yes"
    ) == qss.compiler_output.CompilerOutput(qc, final_logical_to_physical, pulse_sequences=DEFAULT)
    assert backend.compile([qiskit.QuantumCircuit()]) == qss.compiler_output.CompilerOutput(
        [qc], [final_logical_to_physical], pulse_sequences=[DEFAULT]
    )

    with pytest.raises(ValueError, match="'ibmq_jakarta_qpu' is not a valid AQT target."):
        backend.aqt_compile([qc])


@patch("requests.post")
def test_qscout_compile(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    backend = fake_superstaq_provider.get_backend("sandia_qscout_qpu")

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )
    logical_to_physical = {0: 13}
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([list(logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }
    out = backend.compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.final_logical_to_physical == logical_to_physical

    out = backend.compile([qc])
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{0: 13}]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = fake_superstaq_provider.qscout_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]


@patch("requests.post")
def test_compile(mock_post: MagicMock) -> None:
    # Compilation to a simulator (e.g., AWS)
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    backend = provider.get_backend("aws_sv1_simulator")

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc]),
        "final_logical_to_physicals": json.dumps([[(0, 0)]]),
    }
    out = backend.compile([qc], test_options="yes")
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{0: 0}]


def test_target_info(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    target = "ibmq_qasm_simulator"
    backend = fake_superstaq_provider.get_backend(target)
    assert backend.target_info()["backend_name"] == target


@patch("requests.post")
def test_aces(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    backend = provider.get_backend("ss_unconstrained_simulator")

    mock_post.return_value.json = lambda: "id1"
    assert (
        backend.submit_aces(
            qubits=[0, 1],
            shots=100,
            num_circuits=10,
            mirror_depth=5,
            extra_depth=5,
            noise="phase_flip",
            error_prob=0.05,
        )
        == "id1"
    )

    mock_post.return_value.json = lambda: [1] * 51
    assert backend.process_aces("id1") == [1] * 51
