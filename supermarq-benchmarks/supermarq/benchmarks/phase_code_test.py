# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

from typing import cast

import pytest

from supermarq.benchmarks.phase_code import PhaseCode


def test_phase_code_circuit() -> None:
    pc = PhaseCode(3, 1, [1, 1, 1])
    assert len(pc.circuit().all_qubits()) == 5


def test_phase_code_score() -> None:
    pc = PhaseCode(4, 2, [0, 1, 1, 0])
    assert pc.score({"1011010010100": 100}) == 1


def test_invalid_inputs() -> None:
    with pytest.raises(
        ValueError, match="The length of `phase_state` must match the number of data qubits."
    ):
        PhaseCode(3, 1, [0])

    with pytest.raises(ValueError, match=r"`phase_state` must be a list\[int\]."):
        PhaseCode(3, 1, cast("list[int]", "010"))

    with pytest.raises(ValueError, match="Entries of `phase_state` must be 0, 1 integers."):
        PhaseCode(3, 1, cast("list[int]", ["0", "1", "0"]))
