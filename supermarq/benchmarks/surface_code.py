#!/usr/bin/env python3
import functools
from typing import Dict, Tuple

import cirq


def get_code_cycle(rows: int, cols: int, variant_XZZX: bool = True) -> cirq.Circuit:
    """
    Get a single error correction cycle of the planar surface code.
    Construct the XZZX variant by default, unless 'variant_XZZX is False'.
    """
    assert rows > 0 and cols > 0

    data_qubits = _get_data_qubits(rows, cols)
    ancilla_qubits = _get_ancilla_qubits(rows, cols)

    # function to determine whether a qubit is part of *this* surface code patch
    _is_on_this_patch = functools.partial(_is_on_patch, rows=rows, cols=cols)

    # hadamard transforms to go before/after controlled-parity operations for syndrome extraction
    hadamards = cirq.Moment(cirq.H.on_each(*ancilla_qubits))
    if variant_XZZX:
        # add hadamard transforms on every other data qubit turn this into this into the XZZX code
        hadamards += [
            cirq.H(data_qubit) for data_qubit in data_qubits if _qubit_pauli(data_qubit) == cirq.X
        ]

    # construct a cycle of the surface code
    circuit = cirq.Circuit(hadamards)
    for ancilla in filter(_is_on_this_patch, ancilla_qubits):
        neighbors = [
            neighbor for neighbor in _get_diagonal_neighbors(ancilla) if _is_on_this_patch(neighbor)
        ]
        pauli = _qubit_pauli(ancilla)
        pauli_ops = {qubit: pauli for qubit in neighbors}
        parity_op: cirq.PauliString[cirq.GridQubit] = cirq.PauliString(pauli_ops)
        circuit += cirq.decompose_once(parity_op.controlled_by(ancilla))
    circuit += hadamards

    return circuit


def _get_data_qubits(rows: int, cols: int) -> Tuple[cirq.GridQubit, ...]:
    assert rows > 0 and cols > 0
    return tuple(cirq.q(row * 2 + 1, col * 2 + 1) for row in range(rows) for col in range(cols))


def _get_ancilla_qubits(rows: int, cols: int) -> Tuple[cirq.GridQubit, ...]:
    assert rows > 0 and cols > 0
    return tuple(
        qubit
        for row in range(rows + 1)
        for col in range(cols + 1)
        if _is_on_patch(qubit := cirq.q(row * 2, col * 2), rows, cols)
    )


def _is_on_patch(qubit: cirq.GridQubit, rows: int, cols: int) -> bool:
    """Determine whether a given qubit is within a surface code patch."""
    row, col = qubit.row, qubit.col

    if not (0 <= row <= 2 * rows and 0 <= col <= 2 * cols):
        # this qubit is off the charts
        return False

    if row % 2 == col % 2 == 1:
        # this is a data qubit
        return True

    if row % 2 == col % 2 == 0:
        # this is an ancilla qubit

        # check boundaries
        if row == 0 or row == (2 * rows):
            # only X-type ancillas on the top/bottom boundaries
            return 0 < col < (2 * cols) and _qubit_pauli(qubit) == cirq.X
        if col == 0 or col == (2 * cols):
            # only Z-type ancillas on the left/right boundaries
            return 0 < row < (2 * rows) and _qubit_pauli(qubit) == cirq.Z

        # all "interior" ancillas are on the surface code patch
        return True

    # this qubit is not actually a part of the surface code
    return False


def _qubit_pauli(qubit: cirq.GridQubit) -> cirq.Pauli:
    """Determine the pauli operator associated with a qubit."""
    if (qubit.row // 2 % 2) == (qubit.col // 2 % 2):
        return cirq.X
    return cirq.Z


def _get_diagonal_neighbors(
    qubit: cirq.GridQubit,
) -> Tuple[cirq.GridQubit, cirq.GridQubit, cirq.GridQubit, cirq.GridQubit]:
    """Return the diagonal neighbors of a cirq.GridQubit."""
    return (
        cirq.q(qubit.row + 1, qubit.col + 1),
        cirq.q(qubit.row + 1, qubit.col - 1),
        cirq.q(qubit.row - 1, qubit.col + 1),
        cirq.q(qubit.row - 1, qubit.col - 1),
    )


def prepare_logical_state(rows: int, cols: int, variant_XZZX: bool = True) -> cirq.Circuit:
    """
    Prepare a logical state of the planar surface code.
    The method here is a modified version of the method for logical state preparation of the rotated
    surface code in https://arxiv.org/pdf/2002.00362.pdf.
    """
    assert rows > 0 and cols > 0

    circuit = cirq.Circuit()

    # prepare a GHZ state on the first column
    circuit += cirq.H(cirq.q(0, 0))
    for row in range(rows - 1):
        circuit += cirq.CX(cirq.q(row, 0), cirq.q(row + 1, 0))

    # extend the GHZ state column by column
    for col in range(1, cols):
        for row in range(col % 2, rows, 2):
            qubit = cirq.q(row, col)
            circuit += cirq.H(qubit)
            circuit += cirq.CX(qubit, qubit - (0, 1))
            if row > 0:
                circuit += cirq.CX(qubit, qubit - (1, 0))
                circuit += cirq.CX(qubit, qubit - (1, 1))
        if (col % 2) == (rows % 2):
            qubit = cirq.q(rows - 1, col)
            circuit += cirq.H(qubit)
            circuit += cirq.CX(qubit, qubit - (0, 1))

    # transform data qubits to the ones used by the code cycle
    current_qubits = sorted(circuit.all_qubits())
    code_cycle_qubits = sorted(_get_data_qubits(rows, cols))
    qubit_map: Dict[cirq.Qid, cirq.Qid] = dict(zip(current_qubits, code_cycle_qubits))
    circuit = circuit.transform_qubits(qubit_map)

    if variant_XZZX:
        # hadamard transform every other data qubit
        circuit += [cirq.H(qubit) for qubit in code_cycle_qubits if _qubit_pauli(qubit) == cirq.X]
    return circuit
