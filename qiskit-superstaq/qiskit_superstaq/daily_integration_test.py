# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""

import os

import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate, SuperstaqException

import qiskit_superstaq as qss


@pytest.fixture
def provider() -> qss.SuperstaqProvider:
    return qss.SuperstaqProvider()


def test_backends(provider: qss.SuperstaqProvider) -> None:
    result = provider.backends()
    assert provider.get_backend("ibmq_qasm_simulator") in result


def test_ibmq_set_token(provider: qss.SuperstaqProvider) -> None:
    try:
        ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
    except KeyError as key:
        raise KeyError(f"To run the integration tests, please export to {key} a valid IBMQ token")

    assert provider.ibmq_set_token(ibmq_token) == "Your IBMQ account token has been updated"

    with pytest.raises(SuperstaqException, match="IBMQ token is invalid."):
        assert provider.ibmq_set_token("INVALID_TOKEN")


def test_ibmq_compile(provider: qss.SuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.append(qss.AceCR("+-"), [0, 1])
    out = provider.ibmq_compile(qc, target="ibmq_jakarta_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)


def test_acecr_ibmq_compile(provider: qss.SuperstaqProvider) -> None:
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

    out = provider.ibmq_compile(qc, target="ibmq_perth_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)

    out = provider.ibmq_compile(qc, target="ibmq_lagos_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)


def test_aqt_compile(provider: qss.SuperstaqProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(8)
    expected.rz(np.pi / 2, 4)
    expected.rx(np.pi / 2, 4)
    expected.rz(np.pi / 2, 4)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_aqt_compile_eca(provider: qss.SuperstaqProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    circuit.crx(0.7 * np.pi, 4, 5)

    eca_circuits = provider.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits
    assert len(eca_circuits) == 3
    assert all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in eca_circuits)

    # multiple circuits:
    eca_circuits = provider.aqt_compile([circuit, circuit], num_eca_circuits=3).circuits
    assert len(eca_circuits) == 2
    for circuits in eca_circuits:
        assert len(circuits) == 3
        assert all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in circuits)


@pytest.mark.skip(reason="Won't pass until server issue related to this is fixed")
def test_aqt_compile_eca_regression(provider: qss.SuperstaqProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    circuit.crx(0.7 * np.pi, 4, 5)

    eca_circuits = provider.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits

    # test with same and different seed
    assert (
        eca_circuits == provider.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits
    )
    assert (
        eca_circuits != provider.aqt_compile(circuit, num_eca_circuits=3, random_seed=456).circuits
    )


def test_get_balance(provider: qss.SuperstaqProvider) -> None:
    balance_str = provider.get_balance()
    assert isinstance(balance_str, str)
    assert balance_str.startswith("$")

    assert isinstance(provider.get_balance(pretty_output=False), float)


def test_get_resource_estimate(provider: qss.SuperstaqProvider) -> None:
    circuit1 = qiskit.QuantumCircuit(2)
    circuit1.cnot(0, 1)
    circuit1.h(1)

    resource_estimate = provider.resource_estimate(circuit1, "ss_unconstrained_simulator")

    assert resource_estimate == ResourceEstimate(1, 1, 2)

    circuit2 = qiskit.QuantumCircuit(2)
    circuit2.h(1)
    circuit2.cnot(0, 1)
    circuit2.cz(1, 0)

    resource_estimates = provider.resource_estimate(
        [circuit1, circuit2], "ss_unconstrained_simulator"
    )

    assert resource_estimates == [resource_estimate, ResourceEstimate(1, 2, 3)]


def test_qscout_compile(provider: qss.SuperstaqProvider) -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    expected = qiskit.QuantumCircuit(2)
    expected.r(np.pi / 2, -np.pi / 2, 0)
    expected.z(0)
    assert provider.qscout_compile(circuit).circuit == expected
    assert provider.qscout_compile([circuit]).circuits == [expected]
    assert provider.qscout_compile([circuit, circuit]).circuits == [expected, expected]


def test_qscout_compile_swap_mirror(provider: qss.SuperstaqProvider) -> None:
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


def test_cq_compile(provider: qss.SuperstaqProvider) -> None:
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


def test_get_aqt_configs(provider: qss.superstaq_provider.SuperstaqProvider) -> None:
    res = provider.aqt_get_configs()
    assert "pulses" in res
    assert "variables" in res


def test_supercheq(provider: qss.superstaq_provider.SuperstaqProvider) -> None:
    # fmt: off
    files = [
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1],
    ]
    # fmt: on

    num_qubits = 3
    depth = 1
    circuits, fidelities = provider.supercheq(files, num_qubits, depth)
    assert len(circuits) == 32
    assert fidelities.shape == (32, 32)


def test_submit_to_provider_simulators(provider: qss.superstaq_provider.SuperstaqProvider) -> None:

    qc = qiskit.QuantumCircuit(2, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    backends = ["cq_hilbert_simulator", "aws_sv1_simulator", "ibmq_qasm_simulator"]

    for backend in backends:
        job = provider.get_backend(backend).run(qc, shots=1)
        assert job.result().get_counts() == {"11": 1}
