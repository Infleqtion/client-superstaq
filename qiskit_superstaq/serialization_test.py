import qiskit

import qiskit_superstaq


def test_assign_unique_inst_names() -> None:
    inst_0 = qiskit_superstaq.FermionicSWAPGate(0.1)
    inst_1 = qiskit_superstaq.FermionicSWAPGate(0.2)
    inst_2 = qiskit_superstaq.FermionicSWAPGate(0.1)

    circuit = qiskit.QuantumCircuit(4)
    circuit.append(inst_0, [0, 1])
    circuit.append(inst_1, [1, 2])
    circuit.append(inst_2, [2, 0])
    circuit.append(inst_1, [0, 1])
    circuit.rxx(1.1, 0, 1)
    circuit.rxx(1.2, 1, 2)

    expected_inst_names = [
        "fermionic_swap",
        "fermionic_swap_1",
        "fermionic_swap",
        "fermionic_swap_1",
        "rxx",
        "rxx",
    ]

    new_circuit = qiskit_superstaq.serialization._assign_unique_inst_names(circuit)
    assert [inst.name for inst, _, _ in new_circuit] == expected_inst_names


def test_circuit_serialization() -> None:
    circuit_0 = qiskit.QuantumCircuit(3)
    circuit_0.cx(2, 1)
    circuit_0.rz(0.86, 0)

    circuit_1 = qiskit.QuantumCircuit(4)
    circuit_1.append(qiskit_superstaq.AceCR("+-"), [0, 1])
    circuit_1.append(qiskit_superstaq.AceCR("-+"), [1, 2])
    circuit_1.append(qiskit_superstaq.FermionicSWAPGate(0.75), [2, 0])
    circuit_1.append(
        qiskit_superstaq.ParallelGates(
            qiskit_superstaq.FermionicSWAPGate(3.09),
            qiskit.circuit.library.RXXGate(0.1),
        ),
        [1, 2, 0, 3],
    )
    circuit_1.append(
        qiskit_superstaq.ParallelGates(
            qiskit_superstaq.FermionicSWAPGate(3.09),
            qiskit.circuit.library.RXXGate(0.2),
        ),
        [0, 3, 2, 1],
    )
    circuit_1.append(
        qiskit_superstaq.ParallelGates(
            qiskit_superstaq.FermionicSWAPGate(3.09),
            qiskit.circuit.library.RXXGate(0.1),
        ),
        [0, 1, 3, 2],
    )

    serialized_circuit = qiskit_superstaq.serialization.serialize_circuits(circuit_0)
    assert isinstance(serialized_circuit, str)
    assert qiskit_superstaq.serialization.deserialize_circuits(serialized_circuit) == [circuit_0]

    circuits = [circuit_0, circuit_1]
    serialized_circuits = qiskit_superstaq.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert qiskit_superstaq.serialization.deserialize_circuits(serialized_circuits) == circuits
