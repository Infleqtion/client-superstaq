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
from supermarq.benchmarks.vqe_proxy import VQEProxy


def test_vqe_circuit() -> None:
    vqe = VQEProxy(3, 1)
    assert len(vqe.circuit()) == 2
    assert len(vqe.circuit()[0].all_qubits()) == 3


def test_vqe_score() -> None:
    vqe = VQEProxy(3, 1)
    circuits = vqe.circuit()
    probs = [supermarq.simulation.get_ideal_counts(circ) for circ in circuits]
    assert vqe.score(probs) > 0.99
