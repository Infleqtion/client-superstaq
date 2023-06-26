# pylint: disable=missing-function-docstring,missing-class-docstring
import re

import numpy as np
import pytest


def test_validate_target() -> None:
    provider = qss.SuperstaQProvider(api_key="123")
    with pytest.raises(ValueError, match="does not have a valid string format"):
        qss.SuperstaQBackend(provider=provider, target="invalid_target")

    with pytest.raises(ValueError, match="does not have a valid target device type"):
        qss.SuperstaQBackend(provider=provider, target="ibmq_invalid_device")

    with pytest.raises(ValueError, match="does not have a valid target prefix"):
        qss.SuperstaQBackend(provider=provider, target="invalid_test_qpu")


def test_validate_integer_param() -> None:
    # Tests for valid inputs -> Pass
    valid_inputs = [1, 10, "10", 10.0, np.int16(10), 0b1010]
    for input_value in valid_inputs:
        qss.validation.validate_integer_param(input_value)

    # Tests for invalid input -> TypeError
    invalid_inputs = [None, "reps", "{!r}".format(b"invalid"), 1.5, "1.0", {1}, [1, 2, 3], "0b1010"]
    for input_value in invalid_inputs:
        with pytest.raises(TypeError) as msg:
            qss.validation.validate_integer_param(input_value)
        assert re.search(
            re.escape(f"{input_value} cannot be safely cast as an integer."), str(msg.value)
        )

    # Tests for invalid input -> ValueError
    invalid_values = [0, -1]
    for input_value in invalid_values:
        with pytest.raises(
            ValueError,
            match="Must be a positive integer.",
        ):
            qss.validation.validate_integer_param(input_value)
