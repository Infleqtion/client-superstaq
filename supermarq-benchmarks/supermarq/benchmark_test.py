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

from collections.abc import Mapping

import cirq
import pytest
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


@pytest.fixture
def benchmark() -> Benchmark:
    """Simple one-qubit benchmark that creates an equal superposition state.

    Returns:
        A benchmark instance.
    """

    class _TestBenchmark(Benchmark):
        def circuit(self) -> cirq.Circuit:
            """Returns:
            A test cirq circuit of an equal superposition state.
            """
            qubit = cirq.LineQubit(0)
            return cirq.Circuit(cirq.H(qubit), cirq.measure(qubit))

        def score(self, counts: Mapping[str, float]) -> float:
            """Device performace score.

            Args:
                counts: Dictionary(s) containing the measurement counts from execution.

            Returns:
                a normalized [0,1] score reflecting device performance.
            """
            dist = {b: c / sum(counts.values()) for b, c in counts.items()}
            return hellinger_fidelity({"0": 0.5, "1": 0.5}, dist)

    return _TestBenchmark()


def test_benchmark_circuit(benchmark: Benchmark) -> None:
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0)))
    assert benchmark.circuit() == circuit


def test_benchmark_score(benchmark: Benchmark) -> None:
    assert benchmark.score({"0": 50, "1": 50}) == 1
