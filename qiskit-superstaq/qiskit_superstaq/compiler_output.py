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

import importlib.util
import json
import warnings
from collections.abc import Mapping
from typing import Any

import general_superstaq as gss
import qiskit

import qiskit_superstaq as qss

try:
    import qtrl.sequence_utils.readout
except ModuleNotFoundError:
    pass


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
    """A class that stores the results of `qiskit` compiled circuits."""

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
        """Constructs a `CompilerOutput` object.

        Args:
            circuits: A compiled circuit or a list of compiled circuits or a list of list of
                compiled circuits (e.g., if using ECA).
            initial_logical_to_physicals: Dictionary or list of dictionaries specifying initial
                mapping from logical to physical qubits.
            final_logical_to_physicals: Dictionary or list of dictionaries specifying final mapping
                from logical to physical qubits.
            pulse_gate_circuits: Pulse-gate `qiskit.QuantumCircuit` or list thereof specifying the
                pulse compilation, if available (`None` otherwise).
            seq: `qtrl.sequencer.Sequence` pulse sequence if `qtrl` is available locally.
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


def read_json(
    json_dict: Mapping[str, Any], circuits_is_list: bool, api_version: str = "v0.2.0"
) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's compilation endpoints.

    Args:
        json_dict: A JSON dictionary matching the format returned by /compile endpoint.
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).
        api_version: A string indicating the API version.

    Returns:
        A `CompilerOutput` object with the compiled circuit(s) and (if applicable to this target)
        corresponding pulse gate circuit(s).
    """
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

    pulse_start_times = json_dict.get("pulse_start_times", [])
    for circuit, start_times in zip(compiled_circuits, pulse_start_times):
        circuit._op_start_times = start_times

    pulse_gate_circuits = None

    if "pulse_gate_circuits" in json_dict:
        pulse_gate_circuits = qss.deserialize_circuits(json_dict["pulse_gate_circuits"])

        for circuit, start_times in zip(pulse_gate_circuits, pulse_start_times):
            circuit._op_start_times = start_times

    if circuits_is_list:
        return CompilerOutput(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            pulse_gate_circuits=pulse_gate_circuits,
        )
    return CompilerOutput(
        compiled_circuits[0],
        initial_logical_to_physicals[0],
        final_logical_to_physicals[0],
        pulse_gate_circuits=None if pulse_gate_circuits is None else pulse_gate_circuits[0],
    )


def read_json_aqt(
    json_dict: dict[str, str], circuits_is_list: bool, num_eca_circuits: int | None = None
) -> CompilerOutput:
    """Reads out the returned JSON from Superstaq API's AQT compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /aqt_compile endpoint.
        circuits_is_list: Bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False).
        num_eca_circuits: Optional number of logically equivalent random circuits to generate for
            each input circuit.

    Returns:
        A `CompilerOutput` object with the compiled circuit(s). If `qtrl` is available locally,
        the returned object also stores the pulse sequence in the .seq attribute.
    """
    compiled_circuits: list[qiskit.QuantumCircuit] | list[list[qiskit.QuantumCircuit]]
    compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])

    initial_logical_to_physicals_list: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
    )
    initial_logical_to_physicals: list[dict[int, int]] | list[list[dict[int, int]]] = (
        initial_logical_to_physicals_list
    )

    final_logical_to_physicals_list: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["final_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[int, int]] | list[list[dict[int, int]]] = (
        final_logical_to_physicals_list
    )

    seq = None

    if "state_jp" in json_dict:
        if not importlib.util.find_spec("qtrl"):
            warnings.warn(
                "This output only contains compiled circuits. The `qtrl` package must be installed "
                "in order to deserialize compiled pulse sequences.",
                stacklevel=2,
            )
        else:  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs

            def _sequencer_from_state(state: dict[str, Any]) -> qtrl.sequencer.Sequence:
                seq = qtrl.sequencer.Sequence(n_elements=1)
                seq.__setstate__(state)
                seq.compile()
                return seq

            state = gss.serialization.deserialize(json_dict["state_jp"])

            if "readout_jp" in json_dict:
                readout_state = gss.serialization.deserialize(json_dict["readout_jp"])
                readout_seq = _sequencer_from_state(readout_state)

                if "readout_qubits" in json_dict:
                    readout_qubits = json.loads(json_dict["readout_qubits"])
                    readout_seq._readout = qtrl.sequence_utils.readout._ReadoutInfo(
                        readout_seq, readout_qubits, n_readouts=len(compiled_circuits)
                    )

                state["_readout"] = readout_seq

            seq = _sequencer_from_state(state)

    if num_eca_circuits is not None:
        compiled_circuits = [
            compiled_circuits[i : i + num_eca_circuits]
            for i in range(0, len(compiled_circuits), num_eca_circuits)
        ]

        initial_logical_to_physicals = [
            initial_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(initial_logical_to_physicals_list), num_eca_circuits)
        ]
        final_logical_to_physicals = [
            final_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(final_logical_to_physicals_list), num_eca_circuits)
        ]

    if circuits_is_list:
        return CompilerOutput(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            seq=seq,
        )

    return CompilerOutput(
        compiled_circuits[0],
        initial_logical_to_physicals[0],
        final_logical_to_physicals[0],
        seq=seq,
    )


def read_json_qscout(
    json_dict: dict[str, Any], circuits_is_list: bool, num_eca_circuits: int | None = None
) -> CompilerOutput:
    """Reads out the returned JSON from Superstaq API's QSCOUT compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /qscout_compile endpoint.
        circuits_is_list: Bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False).
        num_eca_circuits: Number of logically equivalent random circuits to generate for each
            input circuit.

    Returns:
        A `CompilerOutput` object with the compiled circuit(s) and a list of
        jaqal programs in a string representation.
    """
    compiled_circuits: list[qiskit.QuantumCircuit] | list[list[qiskit.QuantumCircuit]]
    compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])

    initial_logical_to_physicals_list: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
    )
    initial_logical_to_physicals: list[dict[int, int]] | list[list[dict[int, int]]] = (
        initial_logical_to_physicals_list
    )

    final_logical_to_physicals_list: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["final_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[int, int]] | list[list[dict[int, int]]] = (
        final_logical_to_physicals_list
    )

    jaqal_programs: list[str] = json_dict["jaqal_programs"]

    if num_eca_circuits:
        compiled_circuits = [
            compiled_circuits[i : i + num_eca_circuits]
            for i in range(0, len(compiled_circuits), num_eca_circuits)
        ]
        initial_logical_to_physicals = [
            initial_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(initial_logical_to_physicals_list), num_eca_circuits)
        ]
        final_logical_to_physicals = [
            final_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(final_logical_to_physicals_list), num_eca_circuits)
        ]
        jaqal_programs = [
            gss.compiler_output._jaqal_programs_to_subcircuits(
                jaqal_programs[i : i + num_eca_circuits]
            )
            for i in range(0, len(jaqal_programs), num_eca_circuits)
        ]

    if circuits_is_list:
        return CompilerOutput(
            circuits=compiled_circuits,
            initial_logical_to_physicals=initial_logical_to_physicals,
            final_logical_to_physicals=final_logical_to_physicals,
            jaqal_programs=jaqal_programs,
        )

    return CompilerOutput(
        compiled_circuits[0],
        initial_logical_to_physicals[0],
        final_logical_to_physicals[0],
        jaqal_programs=jaqal_programs,
    )
