from typing import Optional, Sequence

import cirq


def validate_cirq_circuits(circuits: object, check_meas: Optional[bool] = False) -> None:
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

    if check_meas:
        circuit_list = [circuits] if isinstance(circuits, cirq.Circuit) else circuits
        for circuit in circuit_list:
            if isinstance(circuit, cirq.Circuit) and not circuit.has_measurements():
                # TODO: only raise if the run method actually requires samples (and not for e.g. a
                # statevector simulation)
                raise ValueError("Circuit has no measurements to sample.")
