import qiskit

import qiskit_superstaq


def test_circuit_serialization() -> None:
    circuits = [qiskit.QuantumCircuit(3), qiskit.QuantumCircuit(3)]
    circuits[0].cx(2, 1)
    circuits[0].rz(1.23, 0)
    circuits[1].swap(0, 1)
    circuits[1].append(qiskit_superstaq.AceCR("-+"), [2, 1])
    circuits[1].append(qiskit_superstaq.FermionicSWAPGate(4.56), [0, 2])

    serialized_circuits = qiskit_superstaq.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert qiskit_superstaq.serialization.deserialize_circuits(serialized_circuits) == circuits
