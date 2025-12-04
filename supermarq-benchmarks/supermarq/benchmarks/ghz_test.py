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
import numpy as np
import pytest
import qiskit.quantum_info

from supermarq.benchmarks.ghz import GHZ


@pytest.mark.parametrize("method", ["ladder", "star", "logdepth"])
@pytest.mark.parametrize("num_qubits", [3, 4, 7])
def test_ghz_circuit(method: str, num_qubits: int) -> None:
    ghz = GHZ(num_qubits, method=method)

    cirq_circuit = ghz.circuit()
    assert cirq_circuit == ghz.cirq_circuit()

    assert cirq.num_qubits(cirq_circuit) == num_qubits
    expected_state_vector = np.zeros(2**num_qubits)
    expected_state_vector[0] = expected_state_vector[-1] = np.sqrt(0.5)

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq_circuit.final_state_vector(ignore_terminal_measurements=True),
        expected_state_vector,
        atol=1e-8,
    )

    qiskit_circuit = ghz.qiskit_circuit()
    assert qiskit_circuit.num_qubits == num_qubits

    qiskit_circuit.remove_final_measurements()
    cirq.testing.assert_allclose_up_to_global_phase(
        qiskit.quantum_info.Statevector(qiskit_circuit).data,
        expected_state_vector,
        atol=1e-8,
    )


def test_ghz_circuit_methods() -> None:
    star = GHZ(8, method="star").circuit()
    ladder = GHZ(8, method="ladder").circuit()
    logdepth = GHZ(8, method="logdepth").circuit()
    assert len(logdepth) < len(ladder) == len(star)


def test_ghz_invalid_method() -> None:
    with pytest.raises(ValueError, match=r"'foo' is not a valid"):
        _ = GHZ(3, method="foo")


def test_ghz_score() -> None:
    ghz = GHZ(3)
    assert ghz.score({"000": 500, "111": 500}) == 1
