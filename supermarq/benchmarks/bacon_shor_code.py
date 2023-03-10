#!/usr/bin/env python3
import cirq


def get_code_cycle(rows: int, cols: int) -> cirq.Circuit:
    """
    Get a single error correction cycle of the Bacon-Shor code.
    References:
    - https://en.wikipedia.org/wiki/Bacon%E2%80%93Shor_code
    - https://arxiv.org/pdf/quant-ph/0610063.pdf
    """
    assert rows > 0 and cols > 0

    ancillas = {
        (pauli, row, col): cirq.NamedQubit(f"{pauli}_{row}_{col}")
        for pauli in [cirq.Z, cirq.X]
        for row in range(rows - 1)
        for col in range(cols - 1)
    }

    hadamards = cirq.Moment(cirq.H.on_each(*ancillas.values()))

    # construct a code cycle
    circuit = cirq.Circuit(hadamards)
    for (pauli, row, col), ancilla in ancillas.items():
        qubit = cirq.GridQubit(row, col)
        step = (0, 1) if pauli == cirq.Z else (1, 0)
        pauli_ops = {qubit: pauli, qubit + step: pauli}
        parity_op: cirq.PauliString[cirq.GridQubit] = cirq.PauliString(pauli_ops)
        circuit += cirq.decompose_once(parity_op.controlled_by(ancilla))
    circuit += hadamards

    return circuit


def prepare_logical_state(rows: int, cols: int) -> cirq.Circuit:
    """Prepare a logical state of the Bacon-Shor code."""
    assert rows > 0 and cols > 0
    circuit = cirq.Circuit()
    for row in range(rows):
        circuit += cirq.H(cirq.GridQubit(row, 0))
        for col in range(cols - 1):
            circuit += cirq.CX(cirq.GridQubit(row, col), cirq.GridQubit(row, col + 1))
    return circuit
