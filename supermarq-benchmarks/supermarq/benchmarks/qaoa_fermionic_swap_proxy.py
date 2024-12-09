"""Definition of the Fermionic SWAP QAOA benchmark within the Supermarq suite."""

from __future__ import annotations

import cirq
import numpy as np

from supermarq.benchmarks.qaoa_vanilla_proxy import QAOAVanillaProxy


class QAOAFermionicSwapProxy(QAOAVanillaProxy):
    """Proxy of a full Quantum Approximate Optimization Algorithm (QAOA) benchmark.

    This benchmark targets MaxCut on a Sherrington-Kirkpatrick (SK) model. Device
    performance is given by the Hellinger fidelity between the experimental output
    distribution and the true distribution obtained via scalable, classical simulation.

    The ansatz for this QAOA problem utilizes the fermionic SWAP network which is able
    to perform all of the required O(N^2) interactions in linear circuit depth. This
    ansatz is especially well-suited to QPU architectures which only support
    nearest-neighbor connectivity. See https://doi.org/10.3390/electronics10141690 for
    an example of this ansatz used in practice.

    When a new instance of this benchmark is created, the ansatz parameters will
    be initialized by:

    #. Generating a random instance of an SK graph

    #. Finding approximately optimal angles (rather than random values)
    """

    def _gen_ansatz(self, gamma: float, beta: float) -> cirq.Circuit:
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()

        # initialize |++++>
        for q in qubits:
            circuit.append(cirq.H(q))

        # Implement the phase-separator unitary with a swap network
        # The covers indicate which qubits will be swapped at each layer
        cover_a = [(idx - 1, idx) for idx in range(1, self.num_qubits, 2)]
        cover_b = [(idx - 1, idx) for idx in range(2, self.num_qubits, 2)]

        # The indices of the virtual map correspond to physical qubits,
        # the value at that index corresponds to the virtual qubit residing there
        virtual_map = np.arange(self.num_qubits)

        for layer in range(self.num_qubits):
            cover = [cover_a, cover_b][layer % 2]
            for pair in cover:
                i, j = pair  # swap physical qubits i and j

                # Get the corresponding weight between the virtual qubits
                v_i = virtual_map[i]
                v_j = virtual_map[j]
                for edge in self.hamiltonian:
                    if v_i == edge[0] and v_j == edge[1]:
                        weight = edge[2]
                phi = gamma * weight

                # Perform the ZZ+SWAP operation
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))
                circuit.append(cirq.rz(2 * phi)(qubits[j]))
                circuit.append(cirq.CNOT(qubits[j], qubits[i]))
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))

                # update the virtual map
                virtual_map[j], virtual_map[i] = virtual_map[i], virtual_map[j]

        # Implement the mixing unitary
        for q in qubits:
            circuit.append(cirq.rx(2 * beta)(q))

        # Measure all qubits
        circuit.append(cirq.measure(*qubits))

        # NOTE: the final qubits in this circuit are in REVERSED order due to the swap network
        return circuit

    def _get_energy_for_bitstring(self, bitstring: str) -> float:
        # Reverse bitstring ordering due to SWAP network
        return super()._get_energy_for_bitstring(bitstring[::-1])
