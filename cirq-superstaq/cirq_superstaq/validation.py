from typing import Sequence

import cirq


def validate_cirq_circuits(circuits: object, require_measurements: bool = False) -> None:
    """Validates that the input is either a single `cirq.Circuit` or a list of `cirq.Circuit`
    instances.

    Args:
        circuits: The circuit(s) to run.
        require_measurements: An optional boolean flag to check if all of the circuits have
            measurements.

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

    if require_measurements:
        circuit_list = [circuits] if isinstance(circuits, cirq.Circuit) else circuits
        for circuit in circuit_list:
            if isinstance(circuit, cirq.Circuit) and not circuit.has_measurements():
                # TODO: only raise if the run method actually requires samples (and not for e.g. a
                # statevector simulation)
                raise ValueError("Circuit has no measurements to sample.")
