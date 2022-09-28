from unittest.mock import MagicMock

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def test_default_options() -> None:
    ss_provider = qss.SuperstaQProvider(api_key="MY_TOKEN")
    device = qss.SuperstaQBackend(
        provider=ss_provider,
        remote_host=gss.API_URL,
        backend="ibmq_qasm_simulator",
    )

    assert qiskit.providers.Options(shots=1000) == device._default_options()


class MockProvider(qss.SuperstaQProvider):
    def __init__(self) -> None:
        self.api_key = "super.tech"


class MockDevice(qss.SuperstaQBackend):
    def __init__(self) -> None:
        super().__init__(MockProvider(), "super.tech", "mock_backend")
        self._provider = MockProvider()


def test_run() -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])
    device = MockDevice()

    mock_client = MagicMock()
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }

    device._provider._client = mock_client

    answer = device.run(circuits=qc, shots=1000)
    expected = qss.SuperstaQJob(device, "job_id")
    assert answer == expected

    with pytest.raises(ValueError, match="Circuit has no measurements to sample"):
        qc.remove_final_measurements()
        device.run(qc, shots=1000)


def test_multi_circuit_run() -> None:
    device = MockDevice()

    qc1 = qiskit.QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = qiskit.QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])

    mock_client = MagicMock()
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }
    device._provider._client = mock_client

    answer = device.run(circuits=[qc1, qc2], shots=1000)
    expected = qss.SuperstaQJob(device, "job_id")

    assert answer == expected


def test_multi_arg_run() -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])
    device = MockDevice()

    mock_client = MagicMock()
    mock_client.create_job.return_value = {
        "job_ids": ["job_id"],
        "status": "ready",
    }

    device._provider._client = mock_client

    answer = device.run(circuits=qc, shots=1000, method="fake_method", options={"test": "123"})
    expected = qss.SuperstaQJob(device, "job_id")
    assert answer == expected


def test_eq() -> None:

    assert MockDevice() != 3

    provider = qss.SuperstaQProvider(api_key="123")

    backend1 = qss.SuperstaQBackend(
        provider=provider, backend="ibmq_qasm_simulator", remote_host=gss.API_URL
    )
    backend2 = qss.SuperstaQBackend(
        provider=provider, backend="ibmq_athens", remote_host=gss.API_URL
    )
    assert backend1 != backend2

    backend3 = qss.SuperstaQBackend(
        provider=provider, backend="ibmq_qasm_simulator", remote_host=gss.API_URL
    )
    assert backend1 == backend3
