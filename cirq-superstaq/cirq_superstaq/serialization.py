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

import importlib.util
import warnings
from collections.abc import Sequence

import cirq
import stimcirq

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
    resolvers = [*SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS, stimcirq.JSON_RESOLVER]
    circuits = cirq.read_json(json_text=serialized_circuits, resolvers=resolvers)
    if isinstance(circuits, cirq.Circuit):
        return [circuits]
    return circuits


def deserialize_qiskit_circuits(
    serialized_qiskit_circuits: str,
    circuits_is_list: bool,
    pulse_durations: Sequence[Sequence[int]] | None = None,
    pulse_start_times: Sequence[Sequence[int]] | None = None,
) -> list[object] | None:
    """Deserializes `qiskit.QuantumCircuit` objects, if possible; otherwise warns the user.

    Args:
        serialized_qiskit_circuits: Qiskit circuits serialized via `qss.serialize_circuits()`.
        circuits_is_list: Whether to refer to "circuits" (plural) or "circuit" (singular) in warning
            messages.
        pulse_durations: A list of lists of pulse durations, where each list contains the durations
            of every op in the corresponding (serialized) circuit.
        pulse_start_times: A list of lists of start times, where each list contains the start times
            of every op in the corresponding (serialized) circuit.

    Returns:
        A list of deserialized `qiskit.QuantumCircuit` objects, or None if the provided circuits
        could not be deserialized.
    """
    if importlib.util.find_spec("qiskit_superstaq"):
        import qiskit  # noqa: PLC0415
        import qiskit_superstaq as qss  # noqa: PLC0415

        try:
            pulse_gate_circuits = qss.deserialize_circuits(serialized_qiskit_circuits)

        except Exception as e:
            s = "s" if circuits_is_list else ""
            warnings.warn(
                f"Your compiled pulse gate circuit{s} could not be deserialized. Please "
                "make sure your qiskit-superstaq installation is up-to-date (by running "
                "`pip install -U qiskit-superstaq`).\n\n"
                "If the problem persists, please let us know at superstaq@infleqtion.com, "
                "or file a report at https://github.com/Infleqtion/client-superstaq/issues "
                "containing the following information (and any other relevant context):\n\n"
                f"cirq-superstaq version: {css.__version__}\n"
                f"qiskit-superstaq version: {qss.__version__}\n"
                f"qiskit version: {qiskit.__version__}\n"
                f"error: {e!r}\n\n"
                f"You can still access your compiled circuit{s} using the .circuit{s} "
                "attribute of this output.",
                stacklevel=2,
            )
        else:
            if pulse_durations and pulse_start_times:
                pulse_gate_circuits = [
                    qss.serialization.insert_times_and_durations(circuit, durations, start_times)
                    for circuit, durations, start_times in zip(
                        pulse_gate_circuits, pulse_durations, pulse_start_times
                    )
                ]
            return pulse_gate_circuits

    else:
        s = "s" if circuits_is_list else ""
        warnings.warn(
            "qiskit-superstaq is required to deserialize compiled pulse gate circuits. You can "
            "install it with `pip install qiskit-superstaq`.\n\n"
            f"You can still access your compiled circuit{s} using the .circuit{s} attribute of "
            "this output.",
            stacklevel=2,
        )

    return None
