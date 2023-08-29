#!/usr/bin/env python3
from typing import Dict, Tuple

import cirq

from supermarq.benchmark import Benchmark


class SurfaceCode(Benchmark):
    """Creates a circuit for syndrome measurement in the rotated surface code.

    Args:
        rows: The number of rows (of data qubits).
        cols: The number of columns (of data qubits).
        variant_XZZX: Whether to run an XZZX variant of the surface code (default: True).
    """

    def __init__(self, num_rows: int, num_cols: int, variant_XZZX: bool = True) -> None:
        if (
            not isinstance(num_rows, int)
            or not isinstance(num_cols, int)
            or num_rows < 1
            or num_cols < 1
        ):
            raise ValueError("Rows and column numbers must be positive integers!")
        self.rows = num_rows
        self.cols = num_cols
        self.variant_XZZX = variant_XZZX

    def circuit(self) -> cirq.Circuit:
        """Prepare a logical state and run one round of syndrome measurements."""
        circuit = self.prepare_logical_state() + self.get_code_cycle()
        return circuit + cirq.measure(*circuit.all_qubits())

    def get_code_cycle(self) -> cirq.Circuit:
        """Generates a single error-correction cycle of the rotated surface code.

        Returns:
            A `cirq.Circuit`.
        """
        data_qubits = self.get_data_qubits()
        ancilla_qubits = self.get_ancilla_qubits()

        # hadamard transforms to go before/after controlled-parity ops for syndrome extraction
        hadamards = cirq.Moment(cirq.H.on_each(*ancilla_qubits))
        if self.variant_XZZX:
            # add hadamard transforms on every other data qubit
            hadamards += [
                cirq.H(data_qubit)
                for data_qubit in data_qubits
                if self.get_qubit_pauli(data_qubit) == cirq.X
            ]

        # construct a cycle of the surface code
        circuit = cirq.Circuit(hadamards)
        for ancilla in filter(self.is_on_patch, ancilla_qubits):
            neighbors = [
                neighbor
                for neighbor in self.get_diagonal_neighbors(ancilla)
                if self.is_on_patch(neighbor)
            ]
            pauli = self.get_qubit_pauli(ancilla)
            pauli_ops = {qubit: pauli for qubit in neighbors}
            parity_op: cirq.PauliString[cirq.GridQubit] = cirq.PauliString(pauli_ops)
            circuit += cirq.decompose_once(parity_op.controlled_by(ancilla))
        circuit += hadamards

        return circuit

    def get_data_qubits(self) -> Tuple[cirq.GridQubit, ...]:
        """The data qubits on this patch."""
        return tuple(
            cirq.GridQubit(row * 2 + 1, col * 2 + 1)
            for row in range(self.rows)
            for col in range(self.cols)
        )

    def get_ancilla_qubits(self) -> Tuple[cirq.GridQubit, ...]:
        """The ancilla qubits on this patch."""
        return tuple(
            qubit
            for row in range(self.rows + 1)
            for col in range(self.cols + 1)
            if self.is_on_patch(qubit := cirq.GridQubit(row * 2, col * 2))
        )

    def is_on_patch(self, qubit: cirq.GridQubit) -> bool:
        """Is the given qubit on this surface code patch?"""
        row, col = qubit.row, qubit.col

        if not (0 <= row <= 2 * self.rows and 0 <= col <= 2 * self.cols):
            # this qubit is off the charts
            return False

        if row % 2 == col % 2 == 1:
            # this is a data qubit
            return True

        if row % 2 == col % 2 == 0:
            # this is an ancilla qubit

            # check boundaries
            if row == 0 or row == (2 * self.rows):
                # only X-type ancillas on the top/bottom boundaries
                return 0 < col < (2 * self.cols) and self.get_qubit_pauli(qubit) == cirq.X
            if col == 0 or col == (2 * self.cols):
                # only Z-type ancillas on the left/right boundaries
                return 0 < row < (2 * self.rows) and self.get_qubit_pauli(qubit) == cirq.Z

            # all "interior" ancillas are on the surface code patch
            return True

        # this qubit is not actually a part of the surface code
        return False

    def prepare_logical_state(self) -> cirq.Circuit:
        """Prepare a logical |0> state of the rotated surface code.

        The method here is a modified version of that in https://arxiv.org/abs/2002.00362
        """
        circuit = cirq.Circuit()

        # prepare a GHZ state on the first column
        circuit += cirq.H(cirq.GridQubit(0, 0))
        for row in range(self.rows - 1):
            circuit += cirq.CX(cirq.GridQubit(row, 0), cirq.GridQubit(row + 1, 0))

        # extend the GHZ state column by column
        for col in range(1, self.cols):
            for row in range(col % 2, self.rows, 2):
                qubit = cirq.GridQubit(row, col)
                circuit += cirq.H(qubit)
                circuit += cirq.CX(qubit, qubit - (0, 1))
                if row > 0:
                    circuit += cirq.CX(qubit, qubit - (1, 0))
                    circuit += cirq.CX(qubit, qubit - (1, 1))
            if (col % 2) == (self.rows % 2):
                qubit = cirq.GridQubit(self.rows - 1, col)
                circuit += cirq.H(qubit)
                circuit += cirq.CX(qubit, qubit - (0, 1))

        # transform data qubits to the ones used by the code cycle
        current_qubits = sorted(circuit.all_qubits())
        code_cycle_qubits = sorted(self.get_data_qubits())
        qubit_map: Dict[cirq.Qid, cirq.Qid] = dict(zip(current_qubits, code_cycle_qubits))
        circuit = circuit.transform_qubits(qubit_map)

        if self.variant_XZZX:
            # hadamard transform every other data qubit
            circuit += [
                cirq.H(qubit)
                for qubit in code_cycle_qubits
                if self.get_qubit_pauli(qubit) == cirq.X
            ]
        return circuit

    @classmethod
    def get_qubit_pauli(cls, qubit: cirq.GridQubit) -> cirq.Pauli:
        """Determine the pauli operator associated with a qubit."""
        if (qubit.row // 2 % 2) == (qubit.col // 2 % 2):
            return cirq.X
        return cirq.Z

    @classmethod
    def get_diagonal_neighbors(
        cls,
        qubit: cirq.GridQubit,
    ) -> Tuple[cirq.GridQubit, cirq.GridQubit, cirq.GridQubit, cirq.GridQubit]:
        """Return the diagonal neighbors of a cirq.GridQubit."""
        return (
            cirq.GridQubit(qubit.row + 1, qubit.col + 1),
            cirq.GridQubit(qubit.row + 1, qubit.col - 1),
            cirq.GridQubit(qubit.row - 1, qubit.col + 1),
            cirq.GridQubit(qubit.row - 1, qubit.col - 1),
        )

    def score(self, counts: Dict[str, float]) -> float:
        """Benchmark score."""
        return NotImplemented
