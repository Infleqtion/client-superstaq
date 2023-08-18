# pylint: disable=missing-function-docstring,missing-class-docstring
from supermarq.benchmarks.ghz import GHZ


def test_ghz_circuit() -> None:
    ghz = GHZ(3)
    assert len(ghz.circuit().all_qubits()) == 3


def test_ghz_score() -> None:
    ghz = GHZ(3)
    assert ghz.score({"000": 500, "111": 500}) == 1
