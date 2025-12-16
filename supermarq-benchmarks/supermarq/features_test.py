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
import qiskit

import supermarq

CIRCUIT = cirq.Circuit(
    cirq.SWAP(*cirq.LineQubit.range(2)),
    cirq.measure(cirq.LineQubit(0)),
    cirq.reset(cirq.LineQubit(0)),
    cirq.measure(*cirq.LineQubit.range(2)),
)


def test_compute_communication() -> None:
    feature = supermarq.features.compute_communication(CIRCUIT)
    assert feature >= 0
    assert feature <= 1


def test_compute_liveness() -> None:
    feature = supermarq.features.compute_liveness(CIRCUIT)
    assert feature >= 0
    assert feature <= 1


def test_compute_parallelism() -> None:
    feature = supermarq.features.compute_parallelism(CIRCUIT)
    assert feature >= 0
    assert feature <= 1

    assert supermarq.features.compute_parallelism(cirq.Circuit()) == 0


def test_compute_measurement() -> None:
    feature = supermarq.features.compute_measurement(CIRCUIT)
    assert feature >= 0
    assert feature <= 1


def test_compute_entanglement() -> None:
    feature = supermarq.features.compute_entanglement(CIRCUIT)
    assert feature >= 0
    assert feature <= 1


def test_compute_depth() -> None:
    qubits = cirq.LineQubit.range(4)
    test_circuit = cirq.Circuit(
        cirq.CX(qubits[0], qubits[1]),
        cirq.CZ(qubits[2], qubits[3]),
        cirq.CX(qubits[1], qubits[2]),
        cirq.CX(qubits[2], qubits[3]),
    )
    test_feature = supermarq.features.compute_depth(test_circuit)
    assert test_feature >= 0
    assert test_feature <= 1

    assert supermarq.features.compute_depth(cirq.Circuit()) == 0


def test_compute_parallelism_with_qiskit() -> None:
    qiskit_circuit = qiskit.QuantumCircuit(2)
    assert supermarq.features.compute_parallelism_with_qiskit(qiskit_circuit) == 0
