# Copyright 2026 Infleqtion
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

from typing import TYPE_CHECKING, Any

import cirq
import general_superstaq as gss

import cirq_superstaq as css

if TYPE_CHECKING:
    import qtrl.sequence_utils.readout


def active_qubit_indices(circuit: cirq.AbstractCircuit) -> list[int]:
    """Returns the indices of the non-idle qubits in a quantum circuit.

    Note:
        The "index" refers to the argument of a `LineQubit` (so e.g. `cirq.LineQubit(5)`
        has index 5 regardless of the total number of qubits in the circuit).

    Args:
        circuit: The input quantum circuit.

    Returns:
        A list of active qubit indices.

    Raises:
        TypeError: If qubit indices are requested for non-line qubits.
    """
    all_qubits: set[cirq.Qid] = set()
    for op in circuit.all_operations():
        if not isinstance(op.gate, css.Barrier):
            all_qubits.update(op.qubits)

    qubit_indices: list[int] = []
    for q in sorted(all_qubits):
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise TypeError("Qubit indices can only be determined for line qubits.")
        qubit_indices.append(int(q))

    return qubit_indices


def measured_qubit_indices(circuit: cirq.AbstractCircuit) -> list[int]:
    """Returns the indices of the measured qubits in a quantum circuit.

    Note:
        The "index" refers to the argument of a `cirq.LineQubit` (so e.g. `cirq.LineQubit(5)`
        has index 5 regardless of the total number of qubits in the circuit).

    Args:
        circuit: The input quantum circuit.

    Returns:
        A list of the measurement qubit indices.

    Raises:
        TypeError: If qubit indices are requested for non-line qubits.
    """
    unrolled_circuit = cirq.unroll_circuit_op(circuit, deep=True, tags_to_check=None)

    measured_qubits: set[cirq.Qid] = set()
    for _, op in unrolled_circuit.findall_operations(cirq.is_measurement):
        measured_qubits.update(op.qubits)

    qubit_indices: set[int] = set()
    for q in measured_qubits:
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise TypeError("Qubit indices can only be determined for line qubits")
        qubit_indices.add(int(q))

    return sorted(qubit_indices)


class CompilerOutput(gss.compiler_output.BaseCompilerOutput[cirq.Circuit, cirq.Qid]):
    """A class that arranges compiled `cirq` circuit information."""

    def __init__(
        self,
        circuits: (cirq.Circuit | list[cirq.Circuit] | list[list[cirq.Circuit]]),
        initial_logical_to_physicals: (
            dict[cirq.Qid, cirq.Qid]
            | list[dict[cirq.Qid, cirq.Qid]]
            | list[list[dict[cirq.Qid, cirq.Qid]]]
        ),
        final_logical_to_physicals: (
            dict[cirq.Qid, cirq.Qid]
            | list[dict[cirq.Qid, cirq.Qid]]
            | list[list[dict[cirq.Qid, cirq.Qid]]]
        ),
        pulse_gate_circuits: object | None = None,
        seq: qtrl.sequencer.Sequence | None = None,
        jaqal_programs: list[str] | None = None,
    ) -> None:
        """Constructs a `CompilerOutput` object for compiled Cirq `circuits`.

        Args:
            circuits: A compiled circuit or a list of compiled circuits or a list of list of
                compiled circuits (e.g., if using ECA).
            initial_logical_to_physicals: Pre-compilation mapping of logical qubits to physical
                qubits.
            final_logical_to_physicals: Post-compilation mapping of logical qubits to physical
                qubits.
            pulse_gate_circuits: Optional pulse-gate `qiskit.QuantumCircuit` or list thereof
                specifying the pulse compilation, if available.
            seq: An optional `qtrl.sequencer.Sequence` pulse sequence if `qtrl` is available
                locally.
            jaqal_programs: The Jaqal programs as individual strings.
        """
        super().__init__(
            circuits=circuits,
            initial_logical_to_physicals=initial_logical_to_physicals,
            final_logical_to_physicals=final_logical_to_physicals,
            pulse_gate_circuits=pulse_gate_circuits,
            seq=seq,
            jaqal_programs=jaqal_programs,
        )

    @staticmethod
    def _get_deserialized_content(
        json_dict: dict[str, Any],
        circuits_is_list: bool,
    ) -> tuple[
        list[cirq.Circuit],
        list[object] | None,
        list[dict[cirq.Qid, cirq.Qid]],
        list[dict[cirq.Qid, cirq.Qid]],
    ]:
        compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
        initial_logical_to_physicals_list: list[dict[cirq.Qid, cirq.Qid]] = list(
            map(dict, cirq.read_json(json_text=json_dict["initial_logical_to_physicals"]))
        )
        final_logical_to_physicals_list: list[dict[cirq.Qid, cirq.Qid]] = list(
            map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
        )

        pulse_gate_circuits = None
        if "pulse_gate_circuits" in json_dict:
            pulse_gate_circuits = css.serialization.deserialize_qiskit_circuits(
                json_dict["pulse_gate_circuits"],
                circuits_is_list,
                pulse_start_times=json_dict.get("pulse_start_times"),
            )

        return (
            compiled_circuits,
            pulse_gate_circuits,
            initial_logical_to_physicals_list,
            final_logical_to_physicals_list,
        )
