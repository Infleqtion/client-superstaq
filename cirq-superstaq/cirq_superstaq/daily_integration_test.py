# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""

from __future__ import annotations

import os

import cirq
import general_superstaq as gss
import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate

import cirq_superstaq as css


@pytest.fixture
def service() -> css.Service:
    """Fixture for cirq_superstaq service.

    Return:
        A cirq_superstaq service instance.
    """
    return css.Service()


def test_ibmq_compile(service: css.Service) -> None:
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        css.AceCRMinusPlus(qubits[0], qubits[1]),
        css.AceCRMinusPlus(qubits[1], qubits[2]),
        css.AceCRMinusPlus(qubits[2], qubits[3]),
    )

    out = service.ibmq_compile(circuit, target="ibmq_brisbane_qpu")
    assert isinstance(out.circuit, cirq.Circuit)
    assert isinstance(out.pulse_gate_circuit, qiskit.QuantumCircuit)
    assert len(out.pulse_gate_circuit.op_start_times) == len(out.pulse_gate_circuit)

    out = service.ibmq_compile([circuit, circuit], target="ibmq_brisbane_qpu")

    assert isinstance(out.circuits, list)
    assert len(out.circuits) == 2
    assert isinstance(out.circuits[1], cirq.Circuit)

    assert isinstance(out.pulse_gate_circuits, list)
    assert len(out.pulse_gate_circuits) == 2
    assert isinstance(out.pulse_gate_circuits[1], qiskit.QuantumCircuit)
    assert len(out.pulse_gate_circuits[1].op_start_times) == len(out.pulse_gate_circuits[1])


def test_ibmq_compile_with_token() -> None:
    service = css.Service(ibmq_token=os.environ["TEST_USER_IBMQ_TOKEN"])
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        css.AceCRMinusPlus(qubits[0], qubits[1]),
        css.AceCRMinusPlus(qubits[1], qubits[2]),
        css.AceCRMinusPlus(qubits[2], qubits[3]),
    )

    out = service.ibmq_compile(circuit, target="ibmq_brisbane_qpu")

    assert isinstance(out.circuit, cirq.Circuit)
    assert isinstance(out.pulse_gate_circuit, qiskit.QuantumCircuit)
    assert len(out.pulse_gate_circuit.op_start_times) == len(out.pulse_gate_circuit)


def test_aqt_compile(service: css.Service) -> None:
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit(cirq.H(qubits[4]))

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        service.aqt_compile(circuit).circuit, circuit, atol=1e-08
    )

    compiled_circuits = service.aqt_compile([circuit]).circuits
    assert isinstance(compiled_circuits, list)
    for compiled_circuit in compiled_circuits:
        assert isinstance(compiled_circuit, cirq.Circuit)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            compiled_circuit, circuit, atol=1e-08
        )
    compiled_circuits = service.aqt_compile([circuit, circuit]).circuits

    assert isinstance(compiled_circuits, list)
    for compiled_circuit in compiled_circuits:
        assert isinstance(compiled_circuit, cirq.Circuit)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            compiled_circuit, circuit, atol=1e-08
        )


def test_aqt_compile_eca(service: css.Service) -> None:
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(4)),
        cirq.CX(cirq.LineQubit(4), cirq.LineQubit(5)) ** 0.7,
    )

    eca_circuits = service.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits
    assert len(eca_circuits) == 3
    assert all(isinstance(circuit, cirq.Circuit) for circuit in eca_circuits)

    # multiple circuits:
    eca_circuits = service.aqt_compile([circuit, circuit], num_eca_circuits=3).circuits
    assert len(eca_circuits) == 2
    for circuits in eca_circuits:
        assert len(circuits) == 3
        assert all(isinstance(circuit, cirq.Circuit) for circuit in circuits)


@pytest.mark.skip(reason="Won't pass until server issue related to this is fixed")
def test_aqt_compile_eca_regression(service: css.Service) -> None:
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(4)),
        cirq.CX(cirq.LineQubit(4), cirq.LineQubit(5)) ** 0.7,
    )
    eca_circuits = service.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits
    # test with same and different seed
    assert (
        eca_circuits == service.aqt_compile(circuit, num_eca_circuits=3, random_seed=123).circuits
    )
    assert (
        eca_circuits != service.aqt_compile(circuit, num_eca_circuits=3, random_seed=456).circuits
    )


def test_get_balance(service: css.Service) -> None:
    balance_str = service.get_balance()
    assert isinstance(balance_str, str)
    assert "credits" in balance_str

    assert isinstance(service.get_balance(pretty_output=False), float)


def test_get_resource_estimate(service: css.Service) -> None:
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)

    circuit1 = cirq.Circuit(cirq.CNOT(q0, q1), cirq.H(q0), cirq.measure(q0))

    resource_estimate = service.resource_estimate(circuit1, "ss_unconstrained_simulator")

    assert resource_estimate == ResourceEstimate(2, 1, 3)

    circuit2 = cirq.Circuit(cirq.H(q1), cirq.CNOT(q0, q1), cirq.CZ(q0, q1), cirq.measure(q1))

    circuits = [circuit1, circuit2]

    resource_estimates = service.resource_estimate(circuits, "ss_unconstrained_simulator")

    assert resource_estimates == [ResourceEstimate(2, 1, 3), ResourceEstimate(2, 2, 4)]


def test_get_targets(service: css.Service) -> None:
    result = service.get_targets()
    filtered_result = service.get_my_targets()
    ibmq_target_info = gss.typing.Target(
        target="ibmq_brisbane_qpu",
        supports_submit=True,
        supports_submit_qubo=False,
        supports_compile=True,
        available=True,
        retired=False,
        accessible=True,
    )
    aqt_target_info = gss.typing.Target(
        target="aqt_keysight_qpu",
        supports_submit=False,
        supports_submit_qubo=False,
        supports_compile=True,
        available=True,
        retired=False,
        accessible=True,
    )

    assert ibmq_target_info in result
    assert aqt_target_info in result
    assert all(target in result for target in filtered_result)


def test_qscout_compile(service: css.Service) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    out = service.qscout_compile(circuit)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        out.circuit, circuit, atol=1e-08
    )
    assert isinstance(out.jaqal_program, str)
    assert "measure_all" in out.jaqal_program

    assert service.qscout_compile([circuit]).circuits == [out.circuit]
    assert service.qscout_compile([circuit, circuit]).circuits == [out.circuit, out.circuit]

    cx_circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1) ** 0.5, cirq.measure(q0, q1))
    out = service.qscout_compile([cx_circuit])
    assert isinstance(out.circuits[0], cirq.Circuit)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        out.circuits[0], cx_circuit, atol=1e-08
    )
    assert isinstance(out.jaqal_programs, list)
    assert isinstance(out.jaqal_programs[0], str)
    assert "MS allqubits[0] allqubits[1]" in out.jaqal_programs[0]


def test_qscout_compile_swap_mirror(service: css.Service) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.SWAP(q0, q1))

    out_qc_swap = cirq.Circuit()

    out = service.qscout_compile(circuit, mirror_swaps=True)
    assert out.circuit == out_qc_swap

    out = service.qscout_compile(circuit, mirror_swaps=False)
    assert cirq.allclose_up_to_global_phase(cirq.unitary(out.circuit), cirq.unitary(circuit))

    num_two_qubit_gates = 0
    for m in out.circuit:
        for op in m:
            if len(op.qubits) > 1:
                num_two_qubit_gates += 1
    assert num_two_qubit_gates == 3


def test_cq_compile(service: css.Service) -> None:
    # We use GridQubits cause CQ's qubits are laid in a grid
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.measure(qubits[0])
    )

    out = service.cq_compile(circuit)
    assert isinstance(out.circuit, cirq.Circuit)


def test_get_aqt_configs(service: css.Service) -> None:
    res = service.aqt_get_configs()
    assert "pulses" in res
    assert "variables" in res


def test_supercheq(service: css.Service) -> None:
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
    circuits, fidelities = service.supercheq(files, num_qubits, depth)
    assert len(circuits) == 32
    assert fidelities.shape == (32, 32)


def test_dfe(service: css.Service) -> None:
    circuit = cirq.Circuit(cirq.H(cirq.q(0)))
    target = "ss_unconstrained_simulator"
    ids = service.submit_dfe(
        rho_1=(circuit, target),
        rho_2=(circuit, target),
        num_random_bases=5,
        shots=1000,
    )
    assert len(ids) == 2

    result = service.process_dfe(ids)
    assert isinstance(result, float)


def test_aces(service: css.Service) -> None:
    noise_model = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(0.1))
    job_id = service.submit_aces(
        target="ss_unconstrained_simulator",
        qubits=[0],
        shots=100,
        num_circuits=10,
        mirror_depth=5,
        extra_depth=7,
        method="noise-sim",
        noise=noise_model,
    )
    result = service.process_aces(job_id)
    assert len(result) == 18


def test_job(service: css.Service) -> None:
    circuit = cirq.Circuit(cirq.measure(cirq.q(0)))
    circuit_alt = cirq.Circuit(cirq.X(cirq.q(0)), cirq.measure(cirq.q(0)))

    job = service.create_job(circuit, target="ibmq_brisbane_qpu", repetitions=10, method="dry-run")
    multi_job = service.create_job(
        [circuit, circuit_alt], target="ibmq_brisbane_qpu", repetitions=10, method="dry-run"
    )

    job_id = job.job_id()  # To test for https://github.com/Infleqtion/client-superstaq/issues/452
    multi_job_id = multi_job.job_id()

    assert job.counts(0) == {"0": 10}
    assert multi_job.counts(0) == {"0": 10}
    assert multi_job.counts(1) == {"1": 10}

    assert job.status() == "Done"
    assert multi_job.status(0) == "Done"
    assert multi_job.status(1) == "Done"

    assert job.job_id() == job_id
    assert multi_job.job_id() == multi_job_id
    assert list(multi_job._job.keys()) == multi_job_id.split(",")

    # Force job to refresh when queried:
    job._job.clear()
    job._job["status"] = "Running"

    # State retrieved from the server should be the same:
    assert job.counts(0) == {"0": 10}
    assert job.status() == "Done"
    assert job.job_id() == job_id


@pytest.mark.parametrize("target", ["cq_sqale_simulator", "aws_sv1_simulator"])
def test_submit_to_provider_simulators(target: str, service: css.Service) -> None:
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))

    job = service.create_job(circuit, repetitions=1, target=target)
    assert job.counts(0) == {"11": 1}


@pytest.mark.skip(reason="Can't be executed when Sqale is set to not accept jobs")
def test_submit_to_sqale_qubit_sorting(service: css.Service) -> None:
    """Regression test for https://github.com/Infleqtion/client-superstaq/issues/776

    Args:
        service: cirq_superstaq service object from fixture.
    """
    target = "cq_sqale_qpu"
    num_qubits = service.target_info(target)["num_qubits"]
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0.0, 24).on(*qubits),
        cirq.rz(np.pi).on(qubits[2]),
        css.ParallelRGate(-np.pi / 2, 0.0, 24).on(*qubits),
        cirq.measure(*qubits),
    )

    job = service.create_job(circuit, repetitions=100, verbatim=True, route=False, target=target)
    counts = job.counts(0)
    assert sum(counts.values()) == 100
    assert max(counts, key=counts.__getitem__) == "001" + ("0" * (num_qubits - 3))


def test_submit_qubo(service: css.Service) -> None:
    test_qubo = {
        (0,): -1,
        (1,): -1,
        (2,): -1,
        (0, 1): 2,
        (1, 2): 2,
    }
    result = service.submit_qubo(test_qubo, target="ss_unconstrained_simulator", repetitions=10)
    assert len(result) == 10
    assert {0: 1, 1: 0, 2: 1} in result
