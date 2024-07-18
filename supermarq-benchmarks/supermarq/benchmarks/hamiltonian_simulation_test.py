# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import supermarq
from supermarq.benchmarks.hamiltonian_simulation import HamiltonianSimulation


def test_hamiltonian_simulation_circuit() -> None:
    hs = HamiltonianSimulation(4, 1, 1)
    assert len(hs.circuit().all_qubits()) == 4
    assert hs.qiskit_circuit().num_qubits == 4


def test_hamiltonian_simulation_score() -> None:
    hs = HamiltonianSimulation(4, 1, 1)
    assert hs._average_magnetization({"1111": 1}, 1) == -1.0
    assert hs._average_magnetization({"0000": 1}, 1) == 1.0
    assert hs.score(supermarq.simulation.get_ideal_counts(hs.circuit())) > 0.99
