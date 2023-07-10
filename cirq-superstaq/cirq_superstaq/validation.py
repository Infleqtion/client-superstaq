from typing import Sequence

import cirq


def validate_cirq_circuits(circuits: object) -> None:
    """Validates that the input is either a single `cirq.Circuit` or a list of `cirq.Circuit`
    instances.

    Args:
        circuits: The circuit(s) to run.

    Raises:
        ValueError: If the input is not a `cirq.Circuit` or a list of `cirq.Circuit` instances.
    """

    if not (
        isinstance(circuits, cirq.Circuit)
        or (
            isinstance(circuits, Sequence)
            and all(isinstance(circuit, cirq.Circuit) for circuit in circuits)
        )
    ):
        raise ValueError(
            "Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
            "sequence of `cirq.Circuit` instances."
        )
