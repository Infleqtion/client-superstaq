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
        raise ValueError("{integer_param} is not a positive integer.")


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
