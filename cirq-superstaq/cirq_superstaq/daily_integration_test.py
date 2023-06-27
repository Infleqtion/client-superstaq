# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""


import os

import cirq
import pytest
from general_superstaq import ResourceEstimate, SuperstaqException

import cirq_superstaq as css


@pytest.fixture
def service() -> css.Service:
    return css.Service()


def test_ibmq_compile(service: css.Service) -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(css.AceCRPlusMinus(qubits[0], qubits[1]))
    out = service.ibmq_compile(circuit, target="ibmq_jakarta_qpu")
    assert isinstance(out.circuit, cirq.Circuit)
    assert out.pulse_sequence is not None


def test_acecr_ibmq_compile(service: css.Service) -> None:
    """Tests ibmq_compile method running without error.

    This test was originally written to make sure compilation to ibmq_casablanca would not fail, but
    IBM has since taken casablanca down.
    """
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        css.AceCRMinusPlus(qubits[0], qubits[1]),
        css.AceCRMinusPlus(qubits[1], qubits[2]),
        css.AceCRMinusPlus(qubits[2], qubits[3]),
    )

    out = service.ibmq_compile(circuit, target="ibmq_jakarta_qpu")
    assert isinstance(out.circuit, cirq.Circuit)
    assert out.pulse_sequence is not None

    out = service.ibmq_compile(circuit, target="ibmq_perth_qpu")
    assert isinstance(out.circuit, cirq.Circuit)
    assert out.pulse_sequence is not None

    out = service.ibmq_compile(circuit, target="ibmq_lagos_qpu")
    assert isinstance(out.circuit, cirq.Circuit)
    assert out.pulse_sequence is not None


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
    assert balance_str.startswith("$")

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


def test_ibmq_set_token(service: css.Service) -> None:
    try:
        ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
    except KeyError as key:
        raise KeyError(f"To run the integration tests, please export to {key} a valid IBMQ token")

    assert service.ibmq_set_token(ibmq_token) == "Your IBMQ account token has been updated"

    with pytest.raises(SuperstaqException, match="IBMQ token is invalid."):
        assert service.ibmq_set_token("INVALID_TOKEN")


def test_get_targets(service: css.Service) -> None:
    result = service.get_targets()
    assert "ibmq_qasm_simulator" in result["compile-and-run"]
    assert "aqt_keysight_qpu" in result["compile-only"]


def test_qscout_compile(service: css.Service) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    compiled_circuit = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(q0),
        cirq.Z(q0) ** -1.0,
        cirq.measure(q0),
    )

    out = service.qscout_compile(circuit)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        out.circuit, compiled_circuit, atol=1e-08
    )
    assert isinstance(out.jaqal_program, str)
    assert "measure_all" in out.jaqal_program

    cx_circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1))
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


def test_job(service: css.Service) -> None:
    circuit = cirq.Circuit(cirq.measure(cirq.q(0)))
    job = service.create_job(circuit, target="ibmq_qasm_simulator", repetitions=10)

    job_id = job.job_id()  # To test for https://github.com/Infleqtion/client-superstaq/issues/452

    assert job.counts() == {"0": 10}
    assert job.status() == "Done"
    assert job.job_id() == job_id

    # Force job to refresh when queried:
    job._job.clear()
    job._job["status"] = "Running"

    # State retrieved from the server should be the same:
    assert job.counts() == {"0": 10}
    assert job.status() == "Done"
    assert job.job_id() == job_id


def test_submit_to_provider_simulators(service: css.Service) -> None:
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))

    targets = ["cq_hilbert_simulator", "aws_sv1_simulator", "ibmq_qasm_simulator"]

    for target in targets:
        job = service.create_job(circuit=circuit, repetitions=1, target=target)
        assert job.counts() == {"11": 1}
