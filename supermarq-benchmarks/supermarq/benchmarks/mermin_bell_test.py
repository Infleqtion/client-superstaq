# pylint: disable=missing-function-docstring,missing-class-docstring
import supermarq
from supermarq.benchmarks.mermin_bell import MerminBell


def test_mermin_bell_circuit() -> None:
    mb = MerminBell(3)
    assert len(mb.circuit().all_qubits()) == 3

    mb = MerminBell(4)
    assert len(mb.circuit().all_qubits()) == 4

    mb = MerminBell(5)
    assert len(mb.circuit().all_qubits()) == 5


def test_mermin_bell_score() -> None:
    mb = MerminBell(3)
    assert mb.score(supermarq.simulation.get_ideal_counts(mb.circuit())) == 1

    mb = MerminBell(4)
    assert mb.score(supermarq.simulation.get_ideal_counts(mb.circuit())) == 1

    mb = MerminBell(5)
    assert mb.score(supermarq.simulation.get_ideal_counts(mb.circuit())) == 1
