# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

from typing import cast

import pytest

from supermarq.benchmarks.bit_code import BitCode


def test_bit_code_circuit() -> None:
    bc = BitCode(3, 1, [1, 1, 1])
    assert len(bc.circuit().all_qubits()) == 5


def test_bit_code_score() -> None:
    bc = BitCode(4, 2, [0, 1, 1, 0])
    assert bc.score({"1011010010100": 100}) == 1


def test_invalid_inputs() -> None:
    with pytest.raises(
        ValueError, match="The length of `bit_state` must match the number of data qubits."
    ):
        BitCode(3, 1, [0])

    with pytest.raises(ValueError, match=r"`bit_state` must be a list\[int\]."):
        BitCode(3, 1, cast("list[int]", "010"))

    with pytest.raises(ValueError, match="Entries of `bit_state` must be 0, 1 integers."):
        BitCode(3, 1, cast("list[int]", ["0", "1", "0"]))
