import re


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
        raise ValueError(f"{integer_param} is not a positive integer.")


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


def validate_noise(noise: object) -> None:
    """Validates that an ACES noise model is valid.

    Args:
        noise: A noise model parameter.

    Raises:
        ValueError: If `noise` is not valid.
    """
    if not (isinstance(noise, tuple) and len(noise) == 2 and isinstance(noise[0], str)):
        raise ValueError(
            f"{noise!r} does not have a valid noise format. Valid noise parameters must be tuples "
            f'of the form `("channel_name", error_prob)`\''
        )

    if noise[0] not in ["symmetric_depolarize", "bit_flip", "phase_flip", "asymmetric_depolarize"]:
        raise ValueError(f"{noise[0]} is not a valid channel.")

    if noise[0] in [
        "symmetric_depolarize",
        "bit_flip",
        "phase_flip",
    ]:
        if not (isinstance(noise[1], (int, float)) and noise[1] >= 0 and noise[1] <= 1):
            raise ValueError(f"{noise[1]} is not a number between 0 and 1.")

    if noise[0] in ["asymmetric_depolarize"]:
        if not (
            isinstance(noise[1], tuple)
            and len(noise[1]) == 3
            and all(isinstance(v, (int, float)) for v in noise[1])
            and sum(noise[1]) <= 1
        ):
            raise ValueError(
                f"{noise[1]} is not of the form (p_x, p_y, p_z) such that p_x + p_y + p_z <= 1."
            )
