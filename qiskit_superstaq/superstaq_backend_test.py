import json
from typing import Any, Dict, List

import pytest
import qiskit
import qiskit_superstaq as qss
import requests


def test_qiskit_to_circuit_json() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    device = qss.superstaq_backend.SuperstaQBackend(
        provider=ss_provider,
        url=qss.API_URL,
        backend="ibmq_qasm_simulator",
    )

    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    qc_barrier = qiskit.QuantumCircuit(2, 2)
    qc_barrier.h(0)
    qc_barrier.barrier()
    qc_barrier.h(0)

    assert device.qiskit_to_circuit_json(qc) == {
        "cirq_type": "Circuit",
        "moments": [
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "GateOperation",
                        "gate": {"cirq_type": "HPowGate", "exponent": 1.0, "global_shift": 0.0},
                        "qubits": [{"cirq_type": "LineQubit", "x": 0}],
                    }
                ],
            },
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "GateOperation",
                        "gate": {"cirq_type": "CXPowGate", "exponent": 1.0, "global_shift": 0.0},
                        "qubits": [
                            {"cirq_type": "LineQubit", "x": 0},
                            {"cirq_type": "LineQubit", "x": 1},
                        ],
                    }
                ],
            },
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "GateOperation",
                        "gate": {
                            "cirq_type": "MeasurementGate",
                            "num_qubits": 1,
                            "key": "c_0",
                            "invert_mask": [],
                        },
                        "qubits": [{"cirq_type": "LineQubit", "x": 0}],
                    },
                    {
                        "cirq_type": "GateOperation",
                        "gate": {
                            "cirq_type": "MeasurementGate",
                            "num_qubits": 1,
                            "key": "c_1",
                            "invert_mask": [],
                        },
                        "qubits": [{"cirq_type": "LineQubit", "x": 1}],
                    },
                ],
            },
        ],
        "device": {"cirq_type": "_UnconstrainedDevice"},
    }
    assert device.qiskit_to_circuit_json(qc_barrier) == {
        "cirq_type": "Circuit",
        "moments": [
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "GateOperation",
                        "gate": {"cirq_type": "HPowGate", "exponent": 1.0, "global_shift": 0.0},
                        "qubits": [{"cirq_type": "LineQubit", "x": 0}],
                    }
                ],
            },
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "TaggedOperation",
                        "sub_operation": {
                            "cirq_type": "GateOperation",
                            "gate": {"cirq_type": "IdentityGate", "num_qubits": 2},
                            "qubits": [
                                {"cirq_type": "LineQubit", "x": 0},
                                {"cirq_type": "LineQubit", "x": 1},
                            ],
                        },
                        "tags": ["barrier"],
                    }
                ],
            },
            {
                "cirq_type": "Moment",
                "operations": [
                    {
                        "cirq_type": "GateOperation",
                        "gate": {"cirq_type": "HPowGate", "exponent": 1.0, "global_shift": 0.0},
                        "qubits": [{"cirq_type": "LineQubit", "x": 0}],
                    }
                ],
            },
        ],
        "device": {"cirq_type": "_UnconstrainedDevice"},
    }


def test_default_options() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider("MY_TOKEN")
    device = qss.superstaq_backend.SuperstaQBackend(
        provider=ss_provider,
        url=qss.API_URL,
        backend="ibmq_qasm_simulator",
    )

    assert qiskit.providers.Options(shots=1000) == device._default_options()


class MockResponse:
    def __init__(self, ids: List[str]) -> None:
        self.content = json.dumps({"ids": ids})

    def json(self) -> Dict:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        pass


class MockBadResponse:
    def __init__(self) -> None:
        self.content = json.dumps({})

    def json(self) -> Dict:
        return {}

    def raise_for_status(self) -> None:
        pass


class MockProvider(qss.superstaq_provider.SuperstaQProvider):
    def __init__(self) -> None:
        self.access_token = "super.tech"


class MockDevice(qss.superstaq_backend.SuperstaQBackend):
    def __init__(self) -> None:
        super().__init__(MockProvider(), "super.tech", "mock_backend")
        self._provider = MockProvider()


def test_run(monkeypatch: Any) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])
    device = MockDevice()

    monkeypatch.setattr(requests, "post", lambda *_, **__: MockResponse(["123abc"]))
    answer = device.run(circuits=qc)
    expected = qss.superstaq_job.SuperstaQJob(device, "123abc")
    assert answer == expected

    monkeypatch.setattr(requests, "post", lambda *_, **__: MockBadResponse())
    with pytest.raises(Exception):
        device.run(circuits=qc)


def test_multi_circuit_run(monkeypatch: Any) -> None:
    device = MockDevice()

    qc1 = qiskit.QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = qiskit.QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])

    monkeypatch.setattr(requests, "post", lambda *_, **__: MockResponse(["123abc", "456efg"]))
    answer = device.run(circuits=[qc1, qc2])
    expected = qss.superstaq_job.SuperstaQJob(device, "123abc,456efg")

    assert answer == expected


def test_eq() -> None:

    assert MockDevice() != 3

    provider = qss.superstaq_provider.SuperstaQProvider(access_token="123")

    backend1 = qss.superstaq_backend.SuperstaQBackend(
        provider=provider, backend="ibmq_qasm_simulator", url=qss.API_URL
    )
    backend2 = qss.superstaq_backend.SuperstaQBackend(
        provider=provider, backend="ibmq_athens", url=qss.API_URL
    )
    assert backend1 != backend2

    backend3 = qss.superstaq_backend.SuperstaQBackend(
        provider=provider, backend="ibmq_qasm_simulator", url=qss.API_URL
    )
    assert backend1 == backend3
