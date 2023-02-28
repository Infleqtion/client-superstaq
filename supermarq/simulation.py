from typing import Dict

import cirq
import numpy as np


def get_ideal_counts(circuit: cirq.Circuit) -> Dict[str, float]:
    ideal_counts = {}
    for i, amplitude in enumerate(circuit.final_state_vector(ignore_terminal_measurements=True)):
        bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return ideal_counts
