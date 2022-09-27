"""Integration checks that run daily (via Github action) between client and prod server."""
import os

import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate, SuperstaQException

import qiskit_superstaq as qss


@pytest.fixture
def provider() -> qss.SuperstaQProvider:
    token = os.environ["TEST_USER_TOKEN"]
    provider = qss.SuperstaQProvider(api_key=token)
    return provider


def test_backends() -> None:
    token = os.environ["TEST_USER_TOKEN"]
    provider = qss.SuperstaQProvider(api_key=token)
    result = provider.backends()
    assert provider.get_backend("ibmq_qasm_simulator") in result
    assert provider.get_backend("d-wave_advantage-system4.1_qpu") in result
    assert provider.get_backend("ionq_ion_qpu") in result


def test_ibmq_set_token() -> None:
    api_token = os.environ["TEST_USER_TOKEN"]
    ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
    provider = qss.SuperstaQProvider(api_key=api_token)
    assert provider.ibmq_set_token(ibmq_token) == "Your IBMQ account token has been updated"

    with pytest.raises(SuperstaQException, match="IBMQ token is invalid."):
        assert provider.ibmq_set_token("INVALID_TOKEN")


def test_ibmq_compile(provider: qss.SuperstaQProvider) -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.append(qss.AceCR("+-"), [0, 1])
    out = provider.ibmq_compile(qc, target="ibmq_jakarta_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)
    assert 800 <= out.pulse_sequence.duration <= 1000  # 896 as of 12/27/2021
    assert out.pulse_sequence.start_time == 0
    assert len(out.pulse_sequence) == 5


def test_acer_non_neighbor_qubits_compile(provider: qss.SuperstaQProvider) -> None:
    qc = qiskit.QuantumCircuit(4)
    qc.append(qss.AceCR("-+"), [0, 1])
    qc.append(qss.AceCR("-+"), [1, 2])
    qc.append(qss.AceCR("-+"), [2, 3])
    out = provider.ibmq_compile(qc, target="ibmq_jakarta_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)
    assert 3000 <= out.pulse_sequence.duration <= 4000  # 3616 as of 6/30/2022
    assert out.pulse_sequence.start_time == 0
    assert len(out.pulse_sequence) == 15


def test_aqt_compile(provider: qss.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(8)
    expected.rz(np.pi / 2, 4)
    expected.rx(np.pi / 2, 4)
    expected.rz(np.pi / 2, 4)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_get_balance(provider: qss.SuperstaQProvider) -> None:
    balance_str = provider.get_balance()
    assert isinstance(balance_str, str)
    assert balance_str.startswith("$")

    assert isinstance(provider.get_balance(pretty_output=False), float)


def test_get_resource_estimate(provider: qss.SuperstaQProvider) -> None:
    circuit1 = qiskit.QuantumCircuit(2)
    circuit1.cnot(0, 1)
    circuit1.h(1)

    resource_estimate = provider.resource_estimate(circuit1, "neutral_atom_qpu")

    assert resource_estimate == ResourceEstimate(1, 1, 2)

    circuit2 = qiskit.QuantumCircuit(2)
    circuit2.h(1)
    circuit2.cnot(0, 1)
    circuit2.cz(1, 0)

    resource_estimates = provider.resource_estimate([circuit1, circuit2], "neutral_atom_qpu")

    assert resource_estimates == [resource_estimate, ResourceEstimate(1, 2, 3)]


def test_qscout_compile(provider: qss.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    expected = qiskit.QuantumCircuit(2)
    expected.u(-np.pi / 2, 0, 0, 0)
    expected.z(0)
    assert provider.qscout_compile(circuit).circuit == expected
    assert provider.qscout_compile([circuit]).circuits == [expected]
    assert provider.qscout_compile([circuit, circuit]).circuits == [expected, expected]


def test_qscout_compile_swap_mirror(provider: qss.SuperstaQProvider) -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.swap(0, 1)

    out_qc_swap = qiskit.QuantumCircuit(2)

    out = provider.qscout_compile(qc, mirror_swaps=True)
    assert out.circuit == out_qc_swap

    out = provider.qscout_compile(qc, mirror_swaps=False)
    op = qiskit.quantum_info.Operator(out.circuit)
    expected_op = qiskit.quantum_info.Operator(qc)
    assert op.equiv(expected_op)

    num_two_qubit_gates = 0
    for _, qbs, _ in out.circuit:
        if len(qbs) > 1:
            num_two_qubit_gates += 1
    assert num_two_qubit_gates == 3


def test_cq_compile(provider: qss.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    assert isinstance(provider.cq_compile(circuit).circuit, qiskit.QuantumCircuit)
    circuits = provider.cq_compile([circuit]).circuits
    assert len(circuits) == 1 and isinstance(circuits[0], qiskit.QuantumCircuit)
    circuits = provider.cq_compile([circuit, circuit]).circuits
    assert (
        len(circuits) == 2
        and isinstance(circuits[0], qiskit.QuantumCircuit)
        and isinstance(circuits[1], qiskit.QuantumCircuit)
    )


def test_get_aqt_configs(provider: qss.superstaq_provider.SuperstaQProvider) -> None:
    res = provider.aqt_get_configs()
    assert "pulses" in res
    assert "variables" in res
