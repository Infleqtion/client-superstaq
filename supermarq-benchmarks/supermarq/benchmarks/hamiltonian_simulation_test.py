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
