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

import json
from typing import TYPE_CHECKING, Any

import general_superstaq as gss
import qiskit

import qiskit_superstaq as qss

if TYPE_CHECKING:
    import qtrl.sequence_utils.readout


def active_qubit_indices(circuit: qiskit.QuantumCircuit) -> list[int]:
    """Returns the indices of the non-idle qubits in the input quantum circuit.

    Args:
        circuit: A `qiskit.QuantumCircuit` circuit.

    Returns:
        A list containing the indices of the non-idle qubits.
    """
    qubit_indices: set[int] = set()

    for inst in circuit:
        if inst.operation.name != "barrier":
            indices = [circuit.find_bit(q).index for q in inst.qubits]
            qubit_indices.update(indices)

    return sorted(qubit_indices)


def measured_qubit_indices(circuit: qiskit.QuantumCircuit) -> list[int]:
    """Returns the indices of the measured qubits in the input quantum circuit.

    Args:
        circuit: A `qiskit.QuantumCircuit` circuit.

    Returns:
        A list containing the indices of the measured qubits.
    """
    measured_qubits: set[qiskit.circuit.Qubit] = set()

    for inst in circuit:
        if isinstance(inst.operation, qiskit.circuit.Measure):
            measured_qubits.update(inst.qubits)

        # Recurse into definition if it involves classical bits
        elif inst.clbits and inst.operation.definition is not None:
            measured_qubits.update(
                inst.qubits[i] for i in measured_qubit_indices(inst.operation.definition)
            )

    return sorted(circuit.find_bit(qubit).index for qubit in measured_qubits)


def classical_bit_mapping(circuit: qiskit.QuantumCircuit) -> dict[int, int]:
    """Returns the index of the (final) measured qubit associated with each classical bit.

    If more than one measurement is assigned to the same classical bit, only the final measurement
    is considered.

    Args:
        circuit: A `qiskit.QuantumCircuit` circuit.

    Returns:
        A dictionary mapping classical bit indices to the indices of the measured qubits.
    """
    clbit_map: dict[qiskit.circuit.Clbit, qiskit.circuit.Qubit] = {}

    for inst in circuit:
        if isinstance(inst.operation, qiskit.circuit.Measure):
            clbit_map[inst.clbits[0]] = inst.qubits[0]

        # Recurse into definition if it involves classical bits
        elif inst.clbits and inst.operation.definition is not None:
            inst_clbit_map = classical_bit_mapping(inst.operation.definition)
            clbit_map.update(
                {inst.clbits[ci]: inst.qubits[qi] for ci, qi in inst_clbit_map.items()}
            )

    return {circuit.find_bit(c).index: circuit.find_bit(q).index for c, q in clbit_map.items()}


class CompilerOutput(gss.BaseCompilerOutput[qiskit.QuantumCircuit, int]):
    """A class that arranges compiled `qiskit` circuit information."""

    def __init__(
        self,
        circuits: (
            qiskit.QuantumCircuit | list[qiskit.QuantumCircuit] | list[list[qiskit.QuantumCircuit]]
        ),
        initial_logical_to_physicals: (
            dict[int, int] | list[dict[int, int]] | list[list[dict[int, int]]]
        ),
        final_logical_to_physicals: (
            dict[int, int] | list[dict[int, int]] | list[list[dict[int, int]]]
        ),
        pulse_gate_circuits: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit] | None = None,
        seq: qtrl.sequencer.Sequence | None = None,
        jaqal_programs: list[str] | None = None,
    ) -> None:
        """Constructs a `CompilerOutput` object for Qiskit `circuits`.

        Args:
            circuits: A compiled circuit or a list of compiled circuits or a list of list of
                compiled circuits (e.g., if using ECA).
            initial_logical_to_physicals: Dictionary or list of dictionaries specifying initial
                mapping from logical to physical qubits.
            final_logical_to_physicals: Dictionary or list of dictionaries specifying final mapping
                from logical to physical qubits.
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


def _generate_compiler_output(
    json_dict: dict[str, Any],
    parser: str,
    circuits_is_list: bool,
    num_eca_circuits: int | None = None,
    api_version: str = gss.API_VERSION,
) -> CompilerOutput:
    if api_version == "v0.2.0":
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
    else:
        serialized_circuits = json.loads(json_dict["qiskit_circuits"])
        compiled_circuits = [
            qss.serialization.deserialize_circuits(circuit)[0] for circuit in serialized_circuits
        ]

    initial_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["final_logical_to_physicals"]))
    )

    if parser == "read_json":
        pulse_start_times = json_dict.get("pulse_start_times", [])
        for circuit, start_times in zip(compiled_circuits, pulse_start_times):
            circuit._op_start_times = start_times

        pulse_gate_circuits = None
        if "pulse_gate_circuits" in json_dict:
            pulse_gate_circuits = qss.deserialize_circuits(json_dict["pulse_gate_circuits"])
            for circuit, start_times in zip(pulse_gate_circuits, pulse_start_times):
                circuit._op_start_times = start_times

        return CompilerOutput.read_json(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            pulse_gate_circuits,
            circuits_is_list,
        )
    if parser == "read_json_qscout":
        jaqal_programs: list[str] = json_dict["jaqal_programs"]
        return CompilerOutput.read_json_qscout(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            jaqal_programs,
            circuits_is_list,
            num_eca_circuits,
        )
    if parser == "read_json_aqt":
        return CompilerOutput.read_json_aqt(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            json_dict,
            circuits_is_list,
            num_eca_circuits,
        )
    raise ValueError(f"Specified parser '{parser}' in an invalid or unavailable parser.")
