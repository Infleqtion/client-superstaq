from unittest import mock

import cirq

import cirq_superstaq.serialization


def test_serialization() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CX(*qubits), cirq_superstaq.ZX(*qubits))

    serialized_circuit = cirq_superstaq.serialization.serialize_circuits(circuit)
    assert isinstance(serialized_circuit, str)
    assert cirq_superstaq.serialization.deserialize_circuits(serialized_circuit) == [circuit]

    circuits = [circuit, circuit]
    serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert cirq_superstaq.serialization.deserialize_circuits(serialized_circuits) == circuits


@mock.patch("cirq.to_json")
def test_removed_device_circuit_atrribute(mock_cirq_to_json: mock.MagicMock) -> None:
    mock_cirq_to_json.return_value = '{\n  "cirq_type": "Circuit",\n  "moments": []\n}'

    circuit = cirq.Circuit()

    serialized_circuit = cirq_superstaq.serialization.serialize_circuits(circuit)
    assert isinstance(serialized_circuit, str)
    assert cirq_superstaq.serialization.deserialize_circuits(serialized_circuit) == [circuit]

    circuits = [circuit, circuit]
    mock_cirq_to_json.return_value = (
        '[\n  {\n    "cirq_type": "Circuit",\n    "moments": []\n  },\n  {\n    '
        '"cirq_type": "Circuit",\n    "moments": []\n  }\n]'
    )
    serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert cirq_superstaq.serialization.deserialize_circuits(serialized_circuits) == circuits
