import cirq

from supermarq.benchmark import Benchmark


class BaconShorCode(Benchmark):
    """Creates a circuit for syndrome measurement in the Bacon-Shor code.

    Args:
    - rows: The number of rows of data qubits
    - cols: The number of columns of data qubits
    """

    def __init__(self, rows: int, cols: int) -> None:
        if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
            raise ValueError("Rows and column numbers must be positive integers!")
        self._rows = rows
        self._cols = cols

    def circuit(self) -> cirq.Circuit:
        """Preparea a logical state and run one round of syndrome measurements."""
        circuit = self.prepare_logical_state() + self.get_code_cycle()
        return circuit + cirq.measure(*circuit.all_qubits())

    def get_code_cycle(self) -> cirq.Circuit:
        """
        Get a single error correction cycle of the Bacon-Shor code.
        References:
        - https://en.wikipedia.org/wiki/Bacon%E2%80%93Shor_code
        - https://arxiv.org/pdf/quant-ph/0610063.pdf
        """
        ancillas = {
            (pauli, row, col): cirq.NamedQubit(f"{pauli}_{row}_{col}")
            for pauli in [cirq.Z, cirq.X]
            for row in range(self._rows - 1)
            for col in range(self._cols - 1)
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

    def prepare_logical_state(self) -> cirq.Circuit:
        """Prepare a logical |0> state of the Bacon-Shor code."""
        circuit = cirq.Circuit()
        for row in range(self._rows):
            circuit += cirq.H(cirq.GridQubit(row, 0))
            for col in range(self._cols - 1):
                circuit += cirq.CX(cirq.GridQubit(row, col), cirq.GridQubit(row, col + 1))
        return circuit

    def score(self, counts: Dict[str, float]) -> float:
        ...
