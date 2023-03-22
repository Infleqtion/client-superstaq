from typing import Dict

import cirq
import numpy as np


def get_ideal_counts(circuit: cirq.Circuit) -> Dict[str, float]:
    """Noiseless statevector simulation.

    Note that the qubits in the returned bitstrings are in big-endian order.
    For example, for a circuit defined on qubits
        q0 ------
        q1 ------
        q2 ------
    the bitstrings are written as `q0q1q2`.

    Args:
        circuit: Input `cirq.Circuit` to be simulated.

    Returns:
        A dictionary with bitstring and probability as the key, value pairs.
    """
    ideal_counts = {}
    for i, amplitude in enumerate(circuit.final_state_vector(ignore_terminal_measurements=True)):
        bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return ideal_counts
