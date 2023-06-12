from typing import Dict, Iterator, List

import cirq
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


class BitCode(Benchmark):
    """Creates a circuit for syndrome measurement in a bit-flip error correcting code.

    Args:
        num_data: The number of data qubits.
        num_rounds: The number of measurement rounds.
        bit_state: A list denoting the state to initialize each data qubit to.

    Returns:
        A `cirq.Circuit` for the bit-flip error correcting code.
    """

    def __init__(self, num_data_qubits: int, num_rounds: int, bit_state: List[int]) -> None:
        if len(bit_state) != num_data_qubits:
            raise ValueError("The length of `bit_state` must match the number of data qubits")
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.bit_state = bit_state

    def _measurement_round_cirq(
        self, qubits: List[cirq.LineQubit], round_idx: int
    ) -> Iterator[cirq.Operation]:
        """Generates `cirq.Operation`s for a single measurement round.

        Args:
            qubits: Circuit qubits, assuming data on even indices and measurement on odd indices.

        Returns:
            A `cirq.Operation` iterator with the operations for a measurement round.
        """
        ancilla_qubits = qubits[1::2]
        for qq in range(1, len(qubits), 2):
            yield cirq.CX(qubits[qq - 1], qubits[qq])
        for qq in range(1, len(qubits), 2):
            yield cirq.CX(qubits[qq + 1], qubits[qq])
        yield cirq.measure(*ancilla_qubits, key=f"mcm{round_idx}")
        yield from cirq.reset_each(*ancilla_qubits)

    def circuit(self) -> cirq.Circuit:
        """Generates bit code circuit.

        Returns:
            A `cirq.Circuit`.
        """
        num_qubits = 2 * self.num_data_qubits - 1
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()

        # Initialize the data qubits
        for i in range(self.num_data_qubits):
            if self.bit_state[i] == 1:
                circuit.append(cirq.X(qubits[2 * i]))

        # Apply measurement rounds
        circuit.append(self._measurement_round_cirq(qubits, i) for i in range(self.num_rounds))

        circuit.append(cirq.measure(*qubits, key="meas_all"))

        return circuit

    def _get_ideal_dist(self) -> Dict[str, float]:
        """Return the ideal probability distribution of `self.circuit()`.

        Since the only allowed initial states for this benchmark are single product states, there
        is a single bitstring that should be measured in the noiseless case.

        Returns:
            Dictionary with measurement results as keys and probabilites as values.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            # parity checks
            ancilla_state += str((self.bit_state[i] + self.bit_state[i + 1]) % 2)
            final_state += str(self.bit_state[i]) + "0"
        else:
            final_state += str(self.bit_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return {"".join(ideal_bitstring): 1.0}

    def score(self, counts: Dict[str, float]) -> float:
        """Compute benchmark score.

        Device performance is given by the Hellinger fidelity between the experimental results and
        the ideal distribution. The ideal is known based on the `bit_state` parameter.

        Args:
            counts: Dictionary containing the measurement counts from running `self.circuit()`.

        Returns:
            A float with the computed score.
        """
        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        experimental_dist = {bitstr: shots / total_shots for bitstr, shots in counts.items()}
        return hellinger_fidelity(ideal_dist, experimental_dist)
