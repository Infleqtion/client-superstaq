"""Integration checks that run daily (via Github action) between client and prod server."""

import os

import numpy as np
import pytest
import qiskit

import qiskit_superstaq


@pytest.fixture
def provider() -> qiskit_superstaq.superstaq_provider.SuperstaQProvider:
    token = os.getenv("TEST_USER_TOKEN")
    token = "93b7b7c9-27a1-4786-a720-d6b3312d59a9"
    provider = qiskit_superstaq.superstaq_provider.SuperstaQProvider(token)
    return provider


def test_aqt_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(1)
    expected.rz(np.pi / 2, 0)
    expected.rx(np.pi / 2, 0)
    expected.rz(np.pi / 2, 0)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]
