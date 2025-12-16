# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator

import cirq
import qiskit
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


class GHZ(Benchmark):
    """Represents the GHZ state preparation benchmark parameterized by the number of qubits n.

    Device performance is based on the Hellinger fidelity between the experimental and ideal
    probability distributions.
    """

    def __init__(self, num_qubits: int, method: str = "ladder") -> None:
        """Initialize a `GHZ` object.

        Args:
            num_qubits: Number of qubits in GHZ circuit.
            method: Circuit construction method to use. Must be "ladder", "star", or "logdepth". The
                "ladder" method uses a linear-depth CNOT ladder, appropriate for nearest-neighbor
                architectures. The "star" method is also linear depth, but with all CNOTs sharing
                the same control qubit. The "logdepth" method uses a log-depth CNOT fanout circuit.
        """
        if method not in ("ladder", "star", "logdepth"):
            raise ValueError(
                f"'{method}' is not a valid GHZ circuit construction method. Valid options are "
                "'ladder', 'star', and 'logdepth'."
            )
        self.num_qubits = num_qubits
        self.method = method

    def circuit(self) -> cirq.Circuit:
        """Generate an n-qubit GHZ cirq circuit.

        Returns:
            A `cirq.Circuit`.
        """
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()
        circuit += cirq.H(qubits[0])

        if self.method == "ladder":
            for i in range(1, self.num_qubits):
                circuit += cirq.CNOT(qubits[i - 1], qubits[i])

        elif self.method == "star":
            for i in range(1, self.num_qubits):
                circuit += cirq.CNOT(qubits[0], qubits[i])

        else:
            for i, j in _fanout(*range(self.num_qubits)):
                circuit += cirq.CNOT(qubits[i], qubits[j])

        circuit += cirq.measure(*qubits)
        return circuit

    def qiskit_circuit(self) -> qiskit.QuantumCircuit:
        """Generate an n-qubit GHZ qiskit circuit.

        Returns:
            A `qiskit.QuantumCircuit`.
        """
        circuit = qiskit.QuantumCircuit(self.num_qubits, self.num_qubits)
        circuit.h(0)

        if self.method == "ladder":
            for i in range(1, self.num_qubits):
                circuit.cx(i - 1, i)

        elif self.method == "star":
            for i in range(1, self.num_qubits):
                circuit.cx(0, i)

        else:
            for i, j in _fanout(*range(self.num_qubits)):
                circuit.cx(i, j)

        for i in range(self.num_qubits):
            circuit.measure(i, i)

        return circuit

    def score(self, counts: dict[str, float]) -> float:
        r"""Compute the Hellinger fidelity between the experimental and ideal results.

        The ideal results are 50% probabilty of measuring the all-zero state and 50% probability
        of measuring the all-one state.

        The formula for the Hellinger fidelity between two distributions p and q is given by
        $(\sum_i{p_i q_i})^2$.

        Args:
            counts: A dictionary containing the measurement counts from circuit execution.

        Returns:
            Hellinger fidelity as a float.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total_shots = sum(counts.values())
        device_dist = {bitstr: count / total_shots for bitstr, count in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)


def _fanout(*qubit_indices: int) -> Iterator[tuple[int, int]]:
    if len(qubit_indices) >= 2:
        cutoff = len(qubit_indices) // 2
        yield qubit_indices[0], qubit_indices[cutoff]
        yield from _fanout(*qubit_indices[:cutoff])
        yield from _fanout(*qubit_indices[cutoff:])
