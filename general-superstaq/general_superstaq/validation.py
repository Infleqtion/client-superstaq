import re
from typing import Dict, Sequence


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
    """Checks that a target contains a valid format, vendor prefix, and device type.

    Args:
        target: A string containing the name of a target device.

    Raises:
        ValueError: If `target` has an invalid format, vendor prefix, or device type.
    """
    vendor_prefixes = [
        "aqt",
        "aws",
        "cq",
        "qtm",
        "ibmq",
        "ionq",
        "oxford",
        "quera",
        "rigetti",
        "sandia",
        "ss",
        "toshiba",
    ]

    target_device_types = ["qpu", "simulator"]

    # Check valid format
    match = re.fullmatch("^([A-Za-z0-9-]+)_([A-Za-z0-9-.]+)_([a-z]+)", target)
    if not match:
        raise ValueError(
            f"{target!r} does not have a valid string format. Valid target strings should be in "
            "the form '<provider>_<device>_<type>', e.g. 'ibmq_lagos_qpu'."
        )

    prefix, _, device_type = match.groups()

    # Check valid prefix
    if prefix not in vendor_prefixes:
        raise ValueError(
            f"{target!r} does not have a valid target prefix. Valid prefixes are: "
            f"{vendor_prefixes}."
        )

    # Check for valid device type
    if device_type not in target_device_types:
        raise ValueError(
            f"{target!r} does not have a valid target device type. Valid device types are: "
            f"{target_device_types}."
        )


def validate_noise_type(noise: Dict[str, object], n_qubits: int) -> None:
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
