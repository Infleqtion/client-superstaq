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

import cirq

import supermarq
from supermarq.benchmarks.qaoa_fermionic_swap_proxy import QAOAFermionicSwapProxy


def test_qaoa_circuit() -> None:
    """Test the circuit generation function."""
    qaoa = QAOAFermionicSwapProxy(4)
    assert len(qaoa.circuit().all_qubits()) == 4
    assert (
        len(
            list(qaoa.circuit().findall_operations(lambda op: isinstance(op.gate, type(cirq.CNOT))))
        )
        == 18
    )


def test_qaoa_score() -> None:
    """Test the score evaluation function."""
    qaoa = QAOAFermionicSwapProxy(4)
    # Reverse bitstring ordering due to SWAP network
    raw_counts = supermarq.simulation.get_ideal_counts(qaoa.circuit())
    ideal_counts = {bitstring[::-1]: probability for bitstring, probability in raw_counts.items()}
    assert qaoa.score({k[::-1]: v for k, v in ideal_counts.items()}) > 0.99
