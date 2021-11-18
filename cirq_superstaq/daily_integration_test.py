"""Integration checks that run daily (via Github action) between client and prod server."""

import os
import textwrap

import cirq
import numpy as np
import pytest

import cirq_superstaq


@pytest.fixture
def service() -> cirq_superstaq.Service:
    token = os.getenv("TEST_USER_TOKEN")
    service = cirq_superstaq.Service(api_key=token)
    return service


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
    assert isinstance(service.get_balance(pretty_output=False), float)


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
