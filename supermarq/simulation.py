import collections

import cirq
import numpy as np


def get_ideal_counts(circuit: cirq.Circuit) -> collections.Counter:
    ideal_counts = {}
    for i, amplitude in enumerate(circuit.final_state_vector()):
        bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return collections.Counter(ideal_counts)
