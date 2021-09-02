import qiskit

import qiskit_superstaq


def test_serialization() -> None:
    obj = {"object": ["to", "serialize"]}
    serialized_obj = qiskit_superstaq.converters.serialize(obj)
    assert isinstance(serialized_obj, str)
    assert qiskit_superstaq.converters.deserialize(serialized_obj) == obj


def test_circuit_serialization() -> None:
    circuits = [qiskit.QuantumCircuit(3), qiskit.QuantumCircuit(2)]
    circuits[0].cx(2, 1)
    circuits[0].cz(0, 1)
    circuits[1].swap(0, 1)
    serialized_circuits = qiskit_superstaq.converters.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert qiskit_superstaq.converters.deserialize_circuits(serialized_circuits) == circuits
