from typing import Dict

import cirq
import collections
import numpy as np
import stimcirq



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


def get_ideal_counts_clifford(circuit: cirq.Circuit) -> Dict[str, float]:
    """Noiseless stabilizer simulation using Stim.

    Args:
        circuit: Input `cirq.Circuit` to be simulated. It is converted to a Stim circuit.

    Returns:
        A dictionary with bitstring and probability as the key, value pairs.
    """
    repetitions = 100000
    qubit_order = sorted(circuit.all_qubits())
    qubit_to_index_dict = {q: i for i, q in enumerate(qubit_order)}

    stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(
            circuit, qubit_to_index_dict=qubit_to_index_dict,
        )
    
    sampler = stim_circuit.compile_sampler(seed=np.random.randint(2**32))
    outcome = sampler.sample(repetitions)
    outcome_counter = collections.Counter(map(tuple, np.asarray(outcome, int)))
    
    probs = {}
    for i in outcome_counter:
        temp = ''
        for bit in i:
            temp+=str(bit)
        probs[temp] = outcome_counter[i]/repetitions

    return probs

