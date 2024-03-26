#!/usr/bin/env python3
from typing import Literal

import cirq
import numpy as np

# import qldpc

from supermarq.benchmark import Benchmark


class SurfaceCode(Benchmark):
    """Creates a circuit for syndrome measurement in the rotated surface code.

    Args:
        rows: The number of rows (of data qubits).
        cols: The number of columns (of data qubits).
        xzzx: Whether to run an XZZX variant of the surface code (default: True).

    Example 5x5 rotated surface code layout:

         ―――     ―――
        | ⋅ |   | ⋅ |
        ○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
     ―――○―――○―――○―――○―――○―――
    | × | ⋅ | × | ⋅ | × |
     ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
     ―――○―――○―――○―――○―――○―――
    | × | ⋅ | × | ⋅ | × |
     ―――○―――○―――○―――○―――○
            | ⋅ |   | ⋅ |
             ―――     ―――

    Here:
    - Circles (○) denote data qubits (of which there are 5×5 = 25 total).
    - Tiles with a cross (×) denote X-type parity checks (12 total).
    - Tiles with a dot (⋅) denote Z-type parity checks (12 total).

    All are indexed by integer (row, column) from the top left corner of the patch above, such that:
    - The top left data qubit is at (1, 1).
    - The next data qubit below (1, 1) is at (3, 1).
    - The leftmost X-type ancilla in the top row is at (0, 4).
    - The upper left Z-type ancilla is at (2, 0).
    Altogether, data qubits are at on odd rows and columns, and ancilla at even rows and columns.

    In the XZZX variant of the surface code, we hadamard-transform every other data qubit.
    """

    def __init__(self, num_rows: int, num_cols: int, xzzx: bool = True) -> None:
        if (
            not isinstance(num_rows, int)
            or not isinstance(num_cols, int)
            or num_rows < 1
            or num_cols < 1
        ):
            raise ValueError("Rows and column numbers must be positive integers!")
        self.rows = num_rows
        self.cols = num_cols
        self.xzzx = xzzx
        # self.code = qldpc.codes.SurfaceCode(num_rows, num_cols, rotated=True)

    def circuit(self) -> cirq.Circuit:
        """Prepare a logical state and run one round of syndrome measurements."""
        circuit = self.prepare_logical_state() + self.get_code_cycle()
        return circuit + cirq.measure(*circuit.all_qubits())

    def get_code_cycle(self) -> cirq.Circuit:
        """Generates a single error-correction cycle of the rotated surface code.

        Returns:
            A `cirq.Circuit`.
        """
        # hadamard transforms to go before/after controlled-parity ops for syndrome extraction
        hadamards = [cirq.H.on_each(*self.get_ancilla_qubits())]
        if self.xzzx:
            # add hadamard transforms on every other data qubit
            hadamards += [
                cirq.H(qubit)
                for qubit in cirq.GridQubit.rect(self.rows, self.cols)
                if self.get_qubit_parity(qubit)
            ]

        # construct a cycle of the surface code
        circuit = cirq.Circuit(hadamards)
        pauli: Literal[cirq.X, cirq.Z]
        for pauli, checks in [(cirq.X, self.code.matrix_x), (cirq.Z, self.code.matrix_z)]:
            checks = checks.reshape((-1, self.rows, self.cols))
            for ancilla, check in zip(self.get_ancilla_qubits(pauli), checks):
                targets = [cirq.GridQubit(row, col) for row, col in zip(np.where(check))]
                pauli_ops = {qubit: pauli for qubit in targets}
                pauli_string: cirq.tring[cirq.GridQubit] = cirq.tring(pauli_ops)
                circuit += pauli_string.controlled_by(ancilla)

        circuit += hadamards
        return circuit

    def get_ancilla_qubits(
        self, pauli: Literal[cirq.X, cirq.Z] | None = None
    ) -> list[cirq.NamedQubit]:
        """Get ancillas on this surface code."""
        if pauli is not None:
            return self.get_ancilla_qubits(cirq.X) + self.get_ancilla_qubits(cirq.Z)
        num_checks = self.code.num_checks_x if pauli == cirq.X else self.code.num_checks_z
        return cirq.NamedQubit.range(num_checks, prefix=str(pauli))

    @classmethod
    def get_qubit_parity(cls, qubit: cirq.GridQubit) -> bool:
        """Should this data qubit be hadamard-transformed in the XZZX code?"""
        return bool((qubit.row + qubit.col) % 2)

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

        if self.xzzx:
            # hadamard transform every other data qubit
            circuit += [
                cirq.H(qubit)
                for qubit in cirq.GridQubit.rect(self.rows, self.cols)
                if self.get_qubit_parity(qubit)
            ]
        return circuit

    def score(self, counts: dict[str, float]) -> float:
        """Benchmark score."""
        return NotImplemented
