"""Integration checks that run daily (via Github action) between client and prod server."""
import os
from datetime import datetime

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


def test_backends(provider: qss.SuperstaQProvider) -> None:
    result = provider.backends()
    assert provider.get_backend("ibmq_qasm_simulator") in result
    if datetime.now().isoweekday() < 6:  # IonQ device is only available on weekdays
        assert provider.get_backend("ionq_ion_qpu") in result


def test_ibmq_set_token(provider: qss.SuperstaQProvider) -> None:
    ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
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


def test_acecr_ibmq_compile(provider: qss.SuperstaQProvider) -> None:
    """Tests ibmq_compile method running without error.

    This test was originally written to make sure compilation to ibmq_casablanca would not fail, but
    IBM has since taken casablanca down.
    """
    qc = qiskit.QuantumCircuit(4)
    qc.append(qss.AceCR("-+"), [0, 1])
    qc.append(qss.AceCR("-+"), [1, 2])
    qc.append(qss.AceCR("-+"), [2, 3])
    out = provider.ibmq_compile(qc, target="ibmq_jakarta_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)
    assert out.pulse_sequence.start_time == 0
    assert len(out.pulse_sequence) == 51

    out = provider.ibmq_compile(qc, target="ibmq_perth_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)
    assert out.pulse_sequence.start_time == 0
    assert len(out.pulse_sequence) == 54

    out = provider.ibmq_compile(qc, target="ibmq_lagos_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)
    assert out.pulse_sequence.start_time == 0
    assert len(out.pulse_sequence) == 61


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


def test_aqt_compile_eca(provider: qss.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)

    eca_circuits = provider.aqt_compile_eca(circuit, num_equivalent_circuits=3).circuits
    assert len(eca_circuits) == 3
    assert all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in eca_circuits)

    eca_circuits = provider.aqt_compile_eca([circuit, circuit], num_equivalent_circuits=3).circuits
    assert len(eca_circuits) == 2
    for circuits in eca_circuits:
        assert len(circuits) == 3
        assert all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in circuits)


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
