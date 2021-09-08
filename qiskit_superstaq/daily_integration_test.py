"""Integration checks that run daily (via Github action) between client and prod server."""

import os

import pytest
import qiskit

import qiskit_superstaq


@pytest.fixture
def provider() -> qiskit_superstaq.superstaq_provider.SuperstaQProvider:
    token = os.environ["TEST_USER_TOKEN"]
    provider = qiskit_superstaq.superstaq_provider.SuperstaQProvider(token)
    return provider


def test_aqt_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(1)
    expected.s(0)
    expected.sx(0)
    expected.s(0)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]
