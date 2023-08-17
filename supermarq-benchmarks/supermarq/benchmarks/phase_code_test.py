# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest

from supermarq.benchmarks.phase_code import PhaseCode


def test_phase_code_circuit() -> None:
    pc = PhaseCode(3, 1, [1, 1, 1])
    assert len(pc.circuit().all_qubits()) == 5


def test_phase_code_score() -> None:
    pc = PhaseCode(4, 2, [0, 1, 1, 0])
    assert pc.score({"1011010010100": 100}) == 1


def test_invalid_size() -> None:
    with pytest.raises(
        ValueError, match="The length of `phase_state` must match the number of data qubits"
    ):
        PhaseCode(3, 1, [0])
