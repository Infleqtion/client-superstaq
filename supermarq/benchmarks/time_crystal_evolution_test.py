# pylint: disable=missing-function-docstring
import supermarq
from supermarq.benchmarks.time_crystal_evolution import TimeCrystalEvolution


def test_time_crystal_circuit() -> None:
    dtc = TimeCrystalEvolution(4, 1)
    assert len(dtc.circuit().all_qubits()) == 4


def test_time_crystal_score() -> None:
    dtc = TimeCrystalEvolution(4, 1)
    assert dtc._average_magnetization({"1111": 1}, 1) == -1.0
    assert dtc._average_magnetization({"0000": 1}, 1) == 1.0
    assert (
        dtc.score(
            {
                key: val * 100
                for key, val in supermarq.simulation.get_ideal_counts(dtc.circuit()).items()
            }
        )
        > 0.99
    )
