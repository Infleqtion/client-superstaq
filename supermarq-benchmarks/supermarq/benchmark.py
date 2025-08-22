from __future__ import annotations

# ruff: noqa: D402
import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cirq


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

    @abc.abstractmethod
    def score(self, counts: Any) -> float:
        """Returns a normalized [0,1] score reflecting device performance.

        Args:
            counts: Dictionary(s) containing the measurement counts from execution.
        """
