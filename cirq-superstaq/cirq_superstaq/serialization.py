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

from collections.abc import Sequence

import cirq

import cirq_superstaq as css

SUPERSTAQ_RESOLVERS = [
    css.ops.qudit_gates.custom_resolver,
    css.ops.qubit_gates.custom_resolver,
]


def serialize_circuits(circuits: cirq.AbstractCircuit | Sequence[cirq.AbstractCircuit]) -> str:
    """Serialize circuit(s) into a json string.

    Args:
        circuits: A `cirq.Circuit` or list of `cirq.Circuits` to be serialized.

    Returns:
        A string representing the serialized circuit(s).
    """
    return cirq.to_json(circuits)


def deserialize_circuits(serialized_circuits: str) -> list[cirq.Circuit]:
    """Deserialize serialized circuit(s).

    Args:
        serialized_circuits: A json string generated via `serialization.serialize_circuit()`.

    Returns:
        The circuit or list of circuits that was serialized.
    """
    resolvers = [*SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS]

    try:
        from stimcirq import JSON_RESOLVER  # noqa: PLC0415
    except ImportError:
        pass
    else:  # pragma: no cover (takes too long to install in CI)
        resolvers.append(JSON_RESOLVER)

    circuits = cirq.read_json(json_text=serialized_circuits, resolvers=resolvers)
    if isinstance(circuits, cirq.Circuit):
        return [circuits]
    return circuits
