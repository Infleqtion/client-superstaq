# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import random
import re

import numpy as np
import pytest

import general_superstaq as gss


def test_validate_target() -> None:
    with pytest.raises(ValueError, match=r"does not have a valid string format"):
        gss.validation.validate_target("invalid_target")

    with pytest.raises(ValueError, match=r"does not have a valid target device type"):
        gss.validation.validate_target("ibmq_invalid_device")


def test_validate_integer_param() -> None:
    # Tests for valid inputs -> Pass
    valid_inputs = [1, 10, "10", 10.0, np.int16(10), 0b1010]
    for input_value in valid_inputs:
        gss.validation.validate_integer_param(input_value)
    gss.validation.validate_integer_param(0, min_val=0)

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
            match=r"is less than the minimum",
        ):
            gss.validation.validate_integer_param(input_value)


def test_validate_noise_type() -> None:
    valid_inputs: list[dict[str, object]] = [
        {"type": "symmetric_depolarize", "params": (0.1,)},
        {"type": "bit_flip", "params": (0.1,)},
        {"type": "phase_flip", "params": (0.1,)},
        {"type": "asymmetric_depolarize", "params": (0.1, 0.1, 0.1)},
    ]
    for input_value in valid_inputs:
        gss.validation.validate_noise_type(input_value, 1)

    with pytest.raises(
        ValueError, match=r"`params` must be a sequence in the dict if `type` is in the dict."
    ):
        gss.validation.validate_noise_type({"type": "test"}, 1)

    with pytest.raises(ValueError, match=r"is not a valid channel."):
        gss.validation.validate_noise_type({"type": "other_channel", "params": (0.1,)}, 1)

    with pytest.raises(ValueError, match='for "bit_flip", and "phase_flip"'):
        gss.validation.validate_noise_type({"type": "bit_flip", "params": (3,)}, 1)

    with pytest.raises(ValueError, match='for "symmetric_depolarize"'):
        gss.validation.validate_noise_type({"type": "symmetric_depolarize", "params": (0.5,)}, 2)

    with pytest.raises(ValueError, match='for "asymmetric_depolarize"'):
        gss.validation.validate_noise_type(
            ({"type": "asymmetric_depolarize", "params": (0.5, 0.5, 0.5)}), 1
        )


def test_validate_bitmap() -> None:
    with pytest.raises(ValueError, match=r"contain the integers 0, 1, or 2"):
        gss.validation.validate_bitmap([[3, 0], [1, 0]])
    rng = np.random.default_rng(random.getrandbits(128))
    with pytest.raises(TypeError, match=r"must be a 2D"):
        gss.validation.validate_bitmap(rng.integers(0, high=3, size=(2, 2, 2)))
    with pytest.raises(TypeError, match=r"must be a square 2D"):
        gss.validation.validate_bitmap(rng.integers(0, high=3, size=(3, 2)))

    valid_bitmaps = (
        [
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2],
        ],
        (
            (1, 0, 1, 2),
            (0, 1, 0, 1),
            (1, 2, 1, 0),
            (0, 1, 0, 1),
        ),
        [
            (1, 1, 2),
            (0, 0, 1),
            (1, 0, 2),
        ],
        np.eye(3, dtype=int),
    )
    for valid_bitmap in valid_bitmaps:
        gss.validation.validate_bitmap(valid_bitmap)


def test_validate_qubo() -> None:
    with pytest.raises(TypeError, match=r"QUBOs must be"):
        gss.validation.validate_qubo("not-a-dict")

    with pytest.raises(TypeError, match=r"not a valid key"):
        gss.validation.validate_qubo({"abc": 123})

    with pytest.raises(TypeError, match=r"must be real numbers"):
        gss.validation.validate_qubo({(1, 2): 12 + 3j})

    with pytest.raises(TypeError, match=r"must be real numbers"):
        gss.validation.validate_qubo({(1, 2): "abc"})

    with pytest.raises(ValueError, match=r"must be quadratic"):
        gss.validation.validate_qubo({(1, 2, 3): 123})

    gss.validation.validate_qubo({(1,): 1.2, (2,): 2.3, (1, 2): -3.4})
    gss.validation.validate_qubo({("a",): 1, ("b",): 2, ("a", "b"): -3})
    gss.validation.validate_qubo({(): 123})
