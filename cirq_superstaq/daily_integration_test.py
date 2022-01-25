"""Integration checks that run daily (via Github action) between client and prod server."""

import os
import textwrap

import cirq
import numpy as np
import pytest
from applications_superstaq import SuperstaQException

import cirq_superstaq


@pytest.fixture
def service() -> cirq_superstaq.Service:
    token = os.getenv("TEST_USER_TOKEN")
    service = cirq_superstaq.Service(api_key=token)
    return service


def test_ibmq_compile(service: cirq_superstaq.Service) -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq_superstaq.AceCRPlusMinus(qubits[0], qubits[1]))
    out = service.ibmq_compile(circuit, target="ibmq_jakarta_qpu")
    assert 800 <= out.duration <= 1000  # 896 as of 12/27/2021
    assert out.start_time == 0
    assert len(out) == 5


def test_aqt_compile(service: cirq_superstaq.Service) -> None:
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit(cirq.H(qubits[4]))
    expected = cirq.Circuit(
        cirq.rz(np.pi / 2)(qubits[4]),
        cirq.rx(np.pi / 2)(qubits[4]),
        cirq.rz(np.pi / 2)(qubits[4]),
    )
    assert service.aqt_compile(circuit).circuit == expected
    assert service.aqt_compile([circuit]).circuits == [expected]
    assert service.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_get_balance(service: cirq_superstaq.Service) -> None:
    balance_str = service.get_balance()
    assert isinstance(balance_str, str)
    assert balance_str.startswith("$")

    assert isinstance(service.get_balance(pretty_output=False), float)


def test_ibmq_set_token() -> None:
    api_token = os.environ["TEST_USER_TOKEN"]
    ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
    service = cirq_superstaq.Service(api_key=api_token)
    assert service.ibmq_set_token(ibmq_token) == "Your IBMQ account token has been updated"

    with pytest.raises(SuperstaQException, match="IBMQ token is invalid."):
        assert service.ibmq_set_token("INVALID_TOKEN")


def test_tsp(service: cirq_superstaq.Service) -> None:
    cities = ["Chicago", "San Francisco", "New York City", "New Orleans"]
    out = service.tsp(cities)
    for city in cities:
        assert city.replace(" ", "+") in out.map_link[0]


def test_get_backends(service: cirq_superstaq.Service) -> None:
    result = service.get_backends()
    assert "ibmq_qasm_simulator" in result["compile-and-run"]
    assert "aqt_keysight_qpu" in result["compile-only"]


def test_qscout_compile(service: cirq_superstaq.Service) -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    compiled_circuit = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(q0),
        cirq.Z(q0) ** -1.0,
        cirq.measure(q0),
    )

    jaqal_program = textwrap.dedent(
        """\
                register allqubits[1]

                prepare_all
                R allqubits[0] -1.5707963267948966 1.5707963267948966
                Rz allqubits[0] -3.141592653589793
                measure_all
                """
    )
    out = service.qscout_compile(circuit)
    assert out.circuit == compiled_circuit
    assert out.jaqal_programs == jaqal_program


def test_cq_compile(service: cirq_superstaq.Service) -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.measure(qubits[0])
    )
    compiled_circuit = cirq.Circuit(
        cirq_superstaq.ParallelRGate(-0.25 * np.pi, 0.5 * np.pi, 2).on(qubits[0], qubits[1]),
        cirq.Z(qubits[0]) ** -1.0,
        cirq.Z(qubits[1]) ** -1.0,
        cirq_superstaq.ParallelRGate(0.25 * np.pi, 0.5 * np.pi, 2).on(qubits[0], qubits[1]),
        cirq.CZ(qubits[0], qubits[1]) ** -1.0,
        cirq.Z(qubits[0]),
        cirq.measure(qubits[0]),
        cirq_superstaq.ParallelRGate(-0.25 * np.pi, 0.5 * np.pi, 2).on(qubits[0], qubits[1]),
        cirq.Z(qubits[1]) ** -1.0,
        cirq_superstaq.ParallelRGate(0.25 * np.pi, 0.5 * np.pi, 2).on(qubits[0], qubits[1]),
    )

    out = service.cq_compile(circuit)
    assert out.circuit == compiled_circuit
