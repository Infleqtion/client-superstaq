"""Integration checks that run daily (via Github action) between client and prod server."""
import os

import numpy as np
import pytest
import qiskit

import qiskit_superstaq


@pytest.fixture
def provider() -> qiskit_superstaq.superstaq_provider.SuperstaQProvider:
    token = os.environ["TEST_USER_TOKEN"]
    provider = qiskit_superstaq.superstaq_provider.SuperstaQProvider(api_key=token)
    return provider


def test_aqt_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(5)
    expected.s(4)
    expected.sx(4)
    expected.s(4)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_qscout_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    expected = qiskit.QuantumCircuit(1)
    expected.u2(-np.pi, np.pi, 0)
    expected.rz(-np.pi, 0)
    assert provider.qscout_compile(circuit).circuit == expected
    assert provider.qscout_compile([circuit]).circuits == [expected]
    assert provider.qscout_compile([circuit, circuit]).circuits == [expected, expected]
