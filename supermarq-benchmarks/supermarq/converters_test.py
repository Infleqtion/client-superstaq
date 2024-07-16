# pylint: disable=missing-function-docstring,missing-class-docstring
# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cirq
import qiskit

import supermarq


def test_cirq_to_qiskit() -> None:
    cirq_circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.CX(cirq.LineQubit(0), cirq.LineQubit(1))
    )
    qiskit_circuit = qiskit.QuantumCircuit(2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    assert supermarq.converters.cirq_to_qiskit(cirq_circuit) == qiskit_circuit


def test_compute_parallelism_with_qiskit() -> None:
    qiskit_circuit = qiskit.QuantumCircuit(2)
    assert supermarq.converters.compute_parallelism_with_qiskit(qiskit_circuit) == 0
