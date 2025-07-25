from __future__ import annotations

import numbers
import re
import warnings
from collections.abc import Mapping, Sequence


def validate_integer_param(integer_param: object, min_val: int = 1) -> None:
    """Validates that `integer_param` is an integer and positive (or above a minimum value).

    Args:
        integer_param: The input parameter to validate.
        min_val: Optional parameter to validate if `integer_param` is greater than `min_val`.

    Raises:
        TypeError: If `integer_param` is not an integer.
        ValueError: If `integer_param` is less than `min_val`.
    """
    if not (
        (hasattr(integer_param, "__int__") and int(integer_param) == integer_param)
        or (isinstance(integer_param, str) and integer_param.isdecimal())
    ):
        raise TypeError(f"{integer_param} cannot be safely cast as an integer.")

    if int(integer_param) < min_val:
        raise ValueError(f"{integer_param} is less than the minimum value ({min_val}).")


def validate_target(target: str) -> None:
    """Checks that `target` conforms to a valid Superstaq format and device type.

    Args:
        target: A string containing the name of a target device.

    Raises:
        ValueError: If `target` has an invalid format or device type.
    """
    target_device_types = ["qpu", "simulator"]

    # Check valid format
    match = re.fullmatch("^([A-Za-z0-9-]+)_([A-Za-z0-9-.]+)_([a-z]+)", target)
    if not match:
        raise ValueError(
            f"{target!r} does not have a valid string format. Valid target strings should be in "
            "the form '<provider>_<device>_<type>', e.g. 'ibmq_brisbane_qpu'."
        )

    _, _, device_type = match.groups()

    # Check for valid device type
    if device_type not in target_device_types:
        raise ValueError(
            f"{target!r} does not have a valid target device type. Valid device types are: "
            f"{target_device_types}."
        )


def validate_noise_type(noise: dict[str, object], n_qubits: int) -> None:
    """Validates that an ACES noise model is valid.

    Args:
        noise: A noise model parameter.
        n_qubits: Number of qubits the noise model is applied to.

    Raises:
        ValueError: If `noise` is not valid.
    """
    noise_type = noise.get("type")
    if not ((params := noise.get("params")) and isinstance(params, Sequence)):
        raise ValueError("`params` must be a sequence in the dict if `type` is in the dict.")

    if noise_type not in [
        "symmetric_depolarize",
        "bit_flip",
        "phase_flip",
        "asymmetric_depolarize",
    ]:
        raise ValueError(f"{noise_type} is not a valid channel.")

    if noise_type in [
        "bit_flip",
        "phase_flip",
    ]:
        if not (
            len(params) == 1
            and isinstance(params[0], (int, float))
            and params[0] >= 0
            and params[0] <= 1
        ):
            raise ValueError(
                f'{params} must be a single number between 0 and 1 for "bit_flip", and '
                f'"phase_flip".'
            )

    if noise_type == "symmetric_depolarize":
        if not (
            len(params) == 1
            and isinstance(params[0], (int, float))
            and params[0] >= 0
            and params[0] <= (1 / (4**n_qubits - 1))
        ):
            raise ValueError(
                f"{params[0]} must be a single number less than 1 / (4^n - 1) for "
                f'"symmetric_depolarize".'
            )

    if noise_type == "asymmetric_depolarize":
        if not (
            len(params) == 3
            and all(isinstance(v, (int, float)) for v in params)
            and sum(params) <= 1
        ):
            raise ValueError(
                f"{params} must be of the form (p_x, p_y, p_z) such that p_x + p_y + p_z <= 1 "
                f'for "asymmetric_depolarize".'
            )


def validate_qubo(qubo: object) -> None:
    """Validates that the input can be converted into a valid QUBO.

    Args:
        qubo: The input value to validate.

    Raises:
        ValueError: If the provided object cannot be converted into a valid QUBO.
    """
    if not isinstance(qubo, Mapping):
        raise ValueError("QUBOs must be provided as dict-like objects.")

    for key, val in qubo.items():
        if not isinstance(key, Sequence) or isinstance(key, str):
            raise ValueError(f"{key!r} is not a valid key for a QUBO.")

        if len(key) > 2:
            raise ValueError(f"QUBOs must be quadratic, but key {key!r} has length {len(key)}.")

        if not isinstance(val, numbers.Real):
            raise ValueError("QUBO values must be real numbers.")


def _validate_ibm_channel(ibm_channel: str) -> None:
    if ibm_channel == "ibm_quantum":
        raise ValueError(
            "The 'ibm_quantum' channel has been deprecated and sunset on July 1st, 2025. Instead, "
            "use 'ibm_quantum_platform' (or equivalently, the older 'ibm_cloud') and the "
            "corresponding channel token.",
        )
    elif ibm_channel == "ibm_cloud":
        warnings.warn(
            "The 'ibm_cloud' channel will be deprecated in the future. Instead, consider using "
            "'ibm_quantum_platform' (the newer version which points to the same channel and works "
            "interchangeably with the same 'ibm_cloud' token and instance).",
            FutureWarning,
            stacklevel=4,
        )
    elif ibm_channel not in ("ibm_cloud", "ibm_quantum_platform"):
        raise ValueError("`ibmq_channel` must be either 'ibm_cloud' or 'ibm_quantum_platform'.")
