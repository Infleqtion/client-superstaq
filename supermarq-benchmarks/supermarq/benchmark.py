from __future__ import annotations

import abc
from typing import Any

import cirq
import qiskit

import supermarq


class Benchmark:
    """Class representing a quantum benchmark application.

    Concrete subclasses must implement the abstract methods ``circuit()`` and
    ``score()``.

    Each instantiation of a `Benchmark` object represents a single, fully defined
    benchmark application. All the relevant parameters for a benchmark should
    be passed in upon creation, and will be used to generate the correct circuit
    and compute the final score.
    """

    @abc.abstractmethod
    def circuit(self) -> cirq.Circuit | list[cirq.Circuit]:
        """Returns the quantum circuit(s) corresponding to the current benchmark parameters."""

    def cirq_circuit(self) -> cirq.Circuit | list[cirq.Circuit]:
        """Returns:
        The cirq circuit(s) corresponding to the current benchmark parameters.
        """
        return self.circuit()

    def qiskit_circuit(self) -> qiskit.QuantumCircuit | list[qiskit.QuantumCircuit]:
        """Returns:
        The qiskit circuit(s) corresponding to the current benchmark parameters.
        """
        cirq_circuit = self.cirq_circuit()
        if isinstance(cirq_circuit, cirq.Circuit):
            return supermarq.converters.cirq_to_qiskit(cirq_circuit)
        return [supermarq.converters.cirq_to_qiskit(c) for c in cirq_circuit]

    @abc.abstractmethod
    def score(self, counts: Any) -> float:
        """Returns a normalized [0,1] score reflecting device performance.

        Args:
            counts: Dictionary(s) containing the measurement counts from execution.
        """
