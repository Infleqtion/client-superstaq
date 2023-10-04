# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""
import os

import general_superstaq as gss
import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate

import qiskit_superstaq as qss


@pytest.fixture
def provider() -> qss.SuperstaqProvider:
    return qss.SuperstaqProvider()


def test_backends(provider: qss.SuperstaqProvider) -> None:
    result = provider.backends()
    assert provider.get_backend("ibmq_qasm_simulator") in result


def test_ibmq_compile(provider: qss.SuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(4)
    qc.append(qss.AceCR("-+"), [0, 1])
    qc.append(qss.AceCR("-+"), [1, 2])
    qc.append(qss.AceCR("-+"), [2, 3])

    out = provider.ibmq_compile(qc, target="ibmq_perth_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)

    out = provider.ibmq_compile(qc, target="ibmq_lagos_qpu")
    assert isinstance(out, qss.compiler_output.CompilerOutput)
    assert isinstance(out.circuit, qiskit.QuantumCircuit)
    assert isinstance(out.pulse_sequence, qiskit.pulse.Schedule)


def test_ibmq_compile_with_token() -> None:
    provider = qss.SuperstaqProvider(ibmq_token=os.environ["TEST_USER_IBMQ_TOKEN"])
    qc = qiskit.QuantumCircuit(4)
    qc.append(qss.AceCR("-+"), [0, 1])
    qc.append(qss.AceCR("-+"), [1, 2])
    qc.append(qss.AceCR("-+"), [2, 3])

    out = provider.ibmq_compile(qc, target="ibmq_perth_qpu")

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

    compiled_circuit = provider.qscout_compile(circuit).circuit
    assert isinstance(compiled_circuit, qiskit.QuantumCircuit)
    assert qiskit.quantum_info.Operator(compiled_circuit) == qiskit.quantum_info.Operator(circuit)

    assert provider.qscout_compile([circuit]).circuits == [compiled_circuit]
    assert provider.qscout_compile([circuit, circuit]).circuits == 2 * [compiled_circuit]


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


def test_dfe(provider: qss.superstaq_provider.SuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    target = "ss_unconstrained_simulator"
    ids = provider.submit_dfe(
        rho_1=(qc, target),
        rho_2=(qc, target),
        num_random_bases=5,
        shots=1000,
    )
    assert len(ids) == 2

    result = provider.process_dfe(ids)
    assert isinstance(result, float)


def test_aces(provider: qss.superstaq_provider.SuperstaqProvider) -> None:
    backend = provider.get_backend("ss_unconstrained_simulator")
    job_id = backend.submit_aces(
        qubits=[0],
        shots=100,
        num_circuits=10,
        mirror_depth=5,
        extra_depth=7,
        method="dry-run",
        noise="bit_flip",
        error_prob=0.1,
    )
    result = backend.process_aces(job_id)
    assert len(result) == 18


@pytest.mark.parametrize(
    "target", ["cq_hilbert_simulator", "aws_sv1_simulator", "ibmq_qasm_simulator"]
)
def test_submit_to_provider_simulators(target: str, provider: qss.SuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    job = provider.get_backend(target).run(qc, shots=1)
    assert job.result().get_counts() == {"11": 1}


@pytest.mark.skip(reason="Can't be executed when Hilbert is set to not accept jobs")
def test_submit_to_hilbert_qubit_sorting(provider: qss.SuperstaqProvider) -> None:
    """Regression test for https://github.com/Infleqtion/client-superstaq/issues/776"""
    backend = provider.get_backend("cq_hilbert_qpu")

    num_qubits = backend.configuration().n_qubits

    gr = qiskit.circuit.library.GR(num_qubits, np.pi / 2, 0)
    grdg = qiskit.circuit.library.GR(num_qubits, -np.pi / 2, 0)

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(gr, range(num_qubits))
    qc.rz(np.pi, 2)
    qc.append(grdg, range(num_qubits))
    qc.measure_all()

    job = backend.run(qc, 100, verbatim=True, route=False)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100
    assert max(counts, key=counts.__getitem__) == ("0" * (num_qubits - 3)) + "100"


def test_submit_qubo(provider: qss.SuperstaqProvider) -> None:
    test_qubo = {(0,): -1, (1,): -1, (2,): -1, (0, 1): 2, (1, 2): 2}
    serialized_result = provider.submit_qubo(
        test_qubo, target="toshiba_bifurcation_qpu", method="dry-run"
    )
    result = gss.qubo.read_json_qubo_result(serialized_result)
    best_result = result[0]
    assert best_result == {0: 1, 1: 0, 2: 1}
