# pylint: disable=missing-function-docstring,missing-class-docstring
import re

import numpy as np
import pytest

import general_superstaq as gss


def test_validate_target() -> None:
    with pytest.raises(ValueError, match="does not have a valid string format"):
        gss.validation.validate_target("invalid_target")

    with pytest.raises(ValueError, match="does not have a valid target device type"):
        gss.validation.validate_target("ibmq_invalid_device")

    with pytest.raises(ValueError, match="does not have a valid target prefix"):
        gss.validation.validate_target("invalid_test_qpu")


def test_validate_integer_param() -> None:
    # Tests for valid inputs -> Pass
    valid_inputs = [1, 10, "10", 10.0, np.int16(10), 0b1010]
    for input_value in valid_inputs:
        gss.validation.validate_integer_param(input_value)

    # Tests for invalid input -> TypeError
    invalid_inputs = [None, "reps", "{!r}".format(b"invalid"), 1.5, "1.0", {1}, [1, 2, 3], "0b1010"]
    for input_value in invalid_inputs:
        with pytest.raises(TypeError) as msg:
            gss.validation.validate_integer_param(input_value)
        assert re.search(
            re.escape(f"{input_value} cannot be safely cast as an integer."), str(msg.value)
        )

    # Tests for invalid input -> ValueError
    invalid_values = [0, -1]
    for input_value in invalid_values:
        with pytest.raises(
            ValueError,
            match="is not a positive integer.",
        ):
            gss.validation.validate_integer_param(input_value)
