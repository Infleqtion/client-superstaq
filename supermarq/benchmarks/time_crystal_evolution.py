from typing import Mapping, Optional

import cirq
import numpy as np

import supermarq
from supermarq.benchmark import Benchmark


class TimeCrystalEvolution(Benchmark):
    """Quantum benchmark based on simulating the evolution of a discrete time crystal (DTC).

    Device performance is based on how closely the experimentally obtained average magnetization
    (i.e., polarization along the Z-axis) matches the noiseless value.

    Based on
        1. 2021 paper observing DTC on Google's QC: https://arxiv.org/abs/2107.13571.
        2. 2023 paper using DTC to characterize IBM QCs: https://arxiv.org/abs/2301.07625.
    """

    def __init__(self, num_qubits: int, num_cycles: int = 1, seed: Optional[int] = None) -> None:
        """Initialize a new time crystal benchmark.

        Args:
            num_qubits: Length of DTC chain.
            num_cycles: Number of times the basic circuit block is repeated.
            seed: Optional integer seed for randomly generated single-qubit rotations.
        """
        self.num_qubits = num_qubits
        self.num_cycles = num_cycles
        self.seed = seed

    def circuit(self) -> cirq.Circuit:
        """Generate a circuit to simulate the evolution of an n-qubit DTC.

        The basic circuit block is defined by the unitary:

        U = exp(-i/2 * sum_i (h_i * Z_i)) *
            exp(-i/4 * sum_i (phi_i * Z_i * Z_{i+1})) *
            exp(-i/2 * pi * g * sum_i (X_i))

        where,
            h_i: are random angles uniformly sampled from [-pi, pi],
            phi_i: are random angles uniformly sampled from [pi/8, 3pi/8],
            g: angle of rotation about the x-axis, set to 0.95.
        And this block is repeated `num_cycles` times.
        """
        np.random.seed(seed=self.seed)
        h_angles = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_qubits)
        phi_angles = np.random.uniform(low=np.pi / 8, high=3 * np.pi / 8, size=self.num_qubits - 1)
        g = 0.95

        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()

        for _ in range(self.num_cycles):
            # Apply single-qubit X rotations
            for qubit in qubits:
                circuit.append(cirq.rx(np.pi * g)(qubit))

            # Coupling terms - CZs with random rotations
            for i, phi in enumerate(phi_angles):
                circuit.append(cirq.ZZPowGate(exponent=phi / (2 * np.pi))(qubits[i], qubits[i + 1]))

            # Apply single-qubit random Z rotations
            for angle, qubit in zip(h_angles, qubits):
                circuit.append(cirq.rz(angle / np.pi)(qubit))

        # End the circuit with measurements of every qubit in the Z-basis
        circuit.append(cirq.measure(*qubits))

        return circuit

    def _average_magnetization(self, result: Mapping[str, float], shots: int) -> float:
        mag = 0.0
        for spin_str, count in result.items():
            spin_int = [1 - 2 * int(s) for s in spin_str]
            mag += (
                sum(spin_int) / len(spin_int)
            ) * count  # <Z> weighted by number of times we saw this bitstring
        average_mag = mag / shots  # normalize by the total number of shots
        return average_mag

    def score(self, counts: Mapping[str, float]) -> float:
        """Compute the average magnetization of the DTC.

        Averages over the measured polarization for each qubit along the Z-axis. Compares the given
        experimental results to the noiseless simulation.

        Args:
            counts: Dictionary of the experimental results. The keys are bitstrings
                represented the measured qubit state, and the values are the number
                of times that state of observed.
        """
        ideal_counts = supermarq.simulation.get_ideal_counts(self.circuit())

        total_shots = int(sum(counts.values()))

        mag_ideal = self._average_magnetization(ideal_counts, 1)
        mag_experimental = self._average_magnetization(counts, total_shots)

        return 1 - abs(mag_ideal - mag_experimental) / 2
