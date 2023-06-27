from typing import Sequence

import qiskit


def validate_qiskit_circuits(circuits: object) -> None:
    """Validates that the input is either a single `qiskit.QuantumCircuit` or a list of
    `qiskit.QuantumCircuit` instances.

    Args:
        circuits: The circuit(s) to run.

    Raises:
        ValueError: If the input is not a `qiskit.QuantumCircuit` or a list of
        `qiskit.QuantumCircuit` instances.
    """
    if not (
        isinstance(circuits, qiskit.QuantumCircuit)
        or (
            isinstance(circuits, Sequence)
            and all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in circuits)
        )
    ):
        raise ValueError(
            "Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a "
            "sequence of `qiskit.QuantumCircuit` instances."
        )
