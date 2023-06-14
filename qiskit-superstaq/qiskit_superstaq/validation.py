import re
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


def validate_integer_param(integer_param: object) -> None:
    """Validates that an input parameter is positive and an integer.

    Args:
        integer_param: An input parameter.

    Raises:
        TypeError: If input is not an integer.
        ValueError: If input is negative.
    """
    if not (
        (hasattr(integer_param, "__int__") and int(integer_param) == integer_param)
        or (isinstance(integer_param, str) and integer_param.isdecimal())
    ):
        raise TypeError(f"{integer_param} cannot be safely cast as an integer.")

    if int(integer_param) <= 0:
        raise ValueError("Must be a positive integer.")


def validate_target(target: str) -> None:
    """Checks that a target contains a valid format, vendor prefix, and device type.
    Args:
        target: A string containing the name of a target backend.

    Raises:
        ValueError: If `target` has an invalid format, vendor prefix, or device type.
    """
    vendor_prefixes = [
        "aqt",
        "aws",
        "cq",
        "hqs",
        "ibmq",
        "ionq",
        "oxford",
        "quera",
        "rigetti",
        "sandia",
        "ss",
    ]

    target_device_types = ["qpu", "simulator"]

    # Check valid format
    match = re.fullmatch("^([A-Za-z0-9-]+)_([A-Za-z0-9-.]+)_([a-z]+)", target)
    if not match:
        raise ValueError(
            f"{target} does not have a valid string format. "
            "Valid target strings should be in the form: "
            "<provider>_<device>_<type>, e.g. ibmq_lagos_qpu."
        )

    prefix, _, device_type = match.groups()

    # Check valid prefix
    if prefix not in vendor_prefixes:
        raise ValueError(
            f"{target} does not have a valid target prefix. "
            f"Valid target prefixes are: {vendor_prefixes}."
        )

    # Check for valid device type
    if device_type not in target_device_types:
        raise ValueError(
            f"{target} does not have a valid target device type. "
            f"Valid target device types are: {target_device_types}."
        )
