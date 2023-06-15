from typing import Dict, Iterator, List

import cirq
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


class PhaseCode(Benchmark):
    """Creates a circuit for syndrome measurement in a phase-flip error correcting code.

    Args:
        num_data: The number of data qubits.
        num_rounds: The number of measurement rounds.
        phase_state: A list of zeros and ones denoting the state to initialize each data
            qubit to. Currently just + or - states. 0 -> +, 1 -> -.

    Returns:
        A `cirq.Circuit` for the phase-flip error correcting code.
    """

    def __init__(self, num_data_qubits: int, num_rounds: int, phase_state: List[int]) -> None:
        if len(phase_state) != num_data_qubits:
            raise ValueError("The length of `phase_state` must match the number of data qubits")
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.phase_state = phase_state

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
        yield from cirq.H.on_each(*qubits)
        for qq in range(1, len(qubits), 2):
            yield cirq.CZ(qubits[qq - 1], qubits[qq])
        for qq in range(1, len(qubits), 2):
            yield cirq.CZ(qubits[qq + 1], qubits[qq])
        yield from cirq.H.on_each(*qubits)
        yield cirq.measure(*ancilla_qubits, key=f"mcm{round_idx}")
        yield from cirq.reset_each(*ancilla_qubits)

    def circuit(self) -> cirq.Circuit:
        """Generates phase code circuit.

        Returns:
            A `cirq.Circuit`.
        """
        num_qubits = 2 * self.num_data_qubits - 1
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()

        # Initialize the data qubits
        for i in range(self.num_data_qubits):
            if self.phase_state[i] == 1:
                circuit.append(cirq.X(qubits[2 * i]))
            circuit.append(cirq.H(qubits[2 * i]))

        # Apply measurement rounds
        circuit.append(self._measurement_round_cirq(qubits, i) for i in range(self.num_rounds))

        # Measure final outcomes in X basis to produce single product state
        for i in range(self.num_data_qubits):
            circuit.append(cirq.H(qubits[2 * i]))

        circuit.append(cirq.measure(*qubits, key="meas_all"))

        return circuit

    def _get_ideal_dist(self) -> Dict[str, float]:
        """Return the ideal probability distribution of `self.circuit()`.

        Since the initial states of the data qubits are either |+> or |->, and we measure the final
        state in the X-basis, the final state is a single product state in the noiseless case.

        Returns:
            Dictionary with measurement results as keys and probabilites as values.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            # parity checks
            ancilla_state += str((self.phase_state[i] + self.phase_state[i + 1]) % 2)
            final_state += str(self.phase_state[i]) + "0"
        else:
            final_state += str(self.phase_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return {"".join(ideal_bitstring): 1.0}

    def score(self, counts: Dict[str, float]) -> float:
        """Compute benchmark score.

        Device performance is given by the Hellinger fidelity between the experimental results and
        the ideal distribution. The ideal is known based on the `phase_state` parameter.

        Args:
            counts: Dictionary containing the measurement counts from running `self.circuit()`.

        Returns:
            A float with the computed score.
        """
        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        experimental_dist = {bitstr: shots / total_shots for bitstr, shots in counts.items()}
        return hellinger_fidelity(ideal_dist, experimental_dist)
