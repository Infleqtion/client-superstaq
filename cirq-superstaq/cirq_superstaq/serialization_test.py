# Copyright 2026 Infleqtion, Inc.
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

from unittest import mock

import cirq
import general_superstaq as gss
import pytest

import cirq_superstaq as css


@mock.patch.dict("sys.modules", {"stimcirq": None})
def test_serialization() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CX(*qubits),
        css.ZX(*qubits),
        cirq.ms(1.23).on(*qubits),
    )

    serialized_circuit_gzip = css.serialization.serialize_circuits(circuit, gzip=True)
    assert isinstance(serialized_circuit_gzip, str)
    assert gss.serialization.str_to_bytes(serialized_circuit_gzip).startswith(b"\x1f\x8b")
    assert css.serialization.deserialize_circuits(serialized_circuit_gzip) == [circuit]

    legacy_serialized_circuit = css.serialization.serialize_circuits(circuit, gzip=False)
    assert legacy_serialized_circuit == cirq.to_json(circuit)
    assert css.serialization.deserialize_circuits(legacy_serialized_circuit) == [circuit]

    circuits = [circuit, circuit]
    serialized_circuits_gzip = css.serialization.serialize_circuits(circuits, gzip=True)
    assert isinstance(serialized_circuits_gzip, str)
    assert css.serialization.deserialize_circuits(serialized_circuits_gzip) == circuits

    legacy_serialized_circuits = cirq.to_json(circuits)
    assert css.serialization.deserialize_circuits(legacy_serialized_circuits) == circuits


def test_serialization_stimcirq() -> None:  # pragma: no cover
    stimcirq = pytest.importorskip("stimcirq", reason="stimcirq not installed")

    circuit = cirq.Circuit(
        cirq.X(cirq.q(0)),
        stimcirq.MeasureAndOrResetGate(
            measure=False,
            reset=True,
            basis="X",
            invert_measure=False,
            key="",
            measure_flip_probability=0,
        ).on(cirq.q(0)),
        stimcirq.DetAnnotation(),
    )

    serialized_circuit = css.serialization.serialize_circuits(circuit)
    assert isinstance(serialized_circuit, str)
    assert css.serialization.deserialize_circuits(serialized_circuit) == [circuit]
