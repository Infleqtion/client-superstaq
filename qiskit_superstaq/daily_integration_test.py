"""Integration checks that run daily (via Github action) between client and prod server."""
import os

import numpy as np
import pytest
import qiskit
from applications_superstaq import SuperstaQException

import qiskit_superstaq


@pytest.fixture
def provider() -> qiskit_superstaq.superstaq_provider.SuperstaQProvider:
    token = os.environ["TEST_USER_TOKEN"]
    provider = qiskit_superstaq.superstaq_provider.SuperstaQProvider(api_key=token)
    return provider


def test_ibmq_set_token() -> None:
    api_token = os.environ["TEST_USER_TOKEN"]
    ibmq_token = os.environ["TEST_USER_IBMQ_TOKEN"]
    provider = qiskit_superstaq.superstaq_provider.SuperstaQProvider(api_key=api_token)
    assert provider.ibmq_set_token(ibmq_token) == "Your IBM Q account token has been updated"

    with pytest.raises(SuperstaQException, match="IBMQ token is invalid."):
        assert provider.ibmq_set_token("INVALID_TOKEN")


def test_aqt_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(8)
    circuit.h(4)
    expected = qiskit.QuantumCircuit(5)
    expected.rz(np.pi / 2, 4)
    expected.rx(np.pi / 2, 4)
    expected.rz(np.pi / 2, 4)
    assert provider.aqt_compile(circuit).circuit == expected
    assert provider.aqt_compile([circuit]).circuits == [expected]
    assert provider.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_qscout_compile(provider: qiskit_superstaq.superstaq_provider.SuperstaQProvider) -> None:
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    expected = qiskit.QuantumCircuit(1)
    expected.u(-np.pi / 2, 0, 0, 0)
    expected.z(0)
    assert provider.qscout_compile(circuit).circuit == expected
    assert provider.qscout_compile([circuit]).circuits == [expected]
    assert provider.qscout_compile([circuit, circuit]).circuits == [expected, expected]
