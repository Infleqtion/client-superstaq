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
import stimcirq

import cirq_superstaq as css


def test_serialization() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        stimcirq.MeasureAndOrResetGate(
            measure=False,
            reset=True,
            basis="X",
            invert_measure=False,
            key="",
            measure_flip_probability=0,
        ).on(cirq.LineQubit(0)),
        cirq.CX(*qubits),
        css.ZX(*qubits),
        cirq.ms(1.23).on(*qubits),
    )

    serialized_circuit = css.serialization.serialize_circuits(circuit)
    assert isinstance(serialized_circuit, str)
    assert css.serialization.deserialize_circuits(serialized_circuit) == [circuit]

    circuits = [circuit, circuit]
    serialized_circuits = css.serialization.serialize_circuits(circuits)
    assert isinstance(serialized_circuits, str)
    assert css.serialization.deserialize_circuits(serialized_circuits) == circuits
