from __future__ import annotations

import importlib.util
import json
import warnings
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

    for inst, qubits, _ in circuit:
        if inst.name != "barrier":
            indices = [circuit.find_bit(q).index for q in qubits]
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

    for inst, qubits, clbits in circuit:
        if isinstance(inst, qiskit.circuit.Measure):
            measured_qubits.update(qubits)

        # Recurse into definition if it involves classical bits
        elif clbits and inst.definition is not None:
            measured_qubits.update(qubits[i] for i in measured_qubit_indices(inst.definition))

    return sorted(circuit.find_bit(qubit).index for qubit in measured_qubits)


def measured_clbit_indices(circuit: qiskit.QuantumCircuit) -> list[int]:
    """Returns the indices of the classical bits in the input quantum circuit.

    Args:
        circuit: A `qiskit.QuantumCircuit` circuit.

    Returns:
        A list containing the indices of the classical bits.
    """
    measured_clbits: set[qiskit.circuit.Clbit] = set()

    for items in circuit:
        inst = items[0]
        clbits = items[2]
        if isinstance(inst, qiskit.circuit.Measure):
            measured_clbits.update(clbits)

        # Recurse into definition
        elif clbits and inst.definition is not None:
            measured_clbits.update(clbits[i] for i in measured_clbit_indices(inst.definition))

    return sorted(circuit.find_bit(bit).index for bit in measured_clbits)


class CompilerOutput:
    """A class that stores the results of compiled circuits."""

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
        pulse_gate_circuits: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit] = None,
        pulse_sequences: qiskit.pulse.Schedule | list[qiskit.pulse.Schedule] | None = None,
        seq: qtrl.sequencer.Sequence | None = None,
        jaqal_programs: str | list[str] | None = None,
        pulse_lists: list[list[list[Any]]] | list[list[list[list[Any]]]] | None = None,
    ) -> None:
        """Constructs a `CompilerOutput` object.

        Args:
            circuits: Compiled circuit or list of compiled circuits.
            initial_logical_to_physicals: Dictionary or list of dictionaries specifying initial
                mapping from logical to physical qubits.
            final_logical_to_physics: Dictionary or list of dictionaries specifying final mapping
                from logical to physical qubits.
            pulse_gate_circuits: Pulse-gate `qiskit.QuantumCircuit` or list thereof specifying the
                pulse compilation.
            pulse_sequences: `qiskit.pulse.Schedule` or list thereof specifying the pulse
                compilation.
            seq: `qtrl.sequencer.Sequence` pulse sequence if `qtrl` is available locally.
            jaqal_programs: Optional string or list of strings specifying Jaqal programs (for
                QSCOUT).
            pulse_lists: Optional list of pulse cycles if `qtrl` is available locally.
        """
        if isinstance(circuits, qiskit.QuantumCircuit):
            self.circuit = circuits
            self.initial_logical_to_physical = initial_logical_to_physicals
            self.final_logical_to_physical = final_logical_to_physicals
            self.pulse_gate_circuit = pulse_gate_circuits
            self.pulse_sequence = pulse_sequences
            self.pulse_list = pulse_lists
            self.jaqal_program = jaqal_programs
        else:
            self.circuits = circuits
            self.initial_logical_to_physicals = initial_logical_to_physicals
            self.final_logical_to_physicals = final_logical_to_physicals
            self.pulse_gate_circuits = pulse_gate_circuits
            self.pulse_sequences = pulse_sequences
            self.pulse_lists = pulse_lists
            self.jaqal_programs = jaqal_programs

        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Checks if this object represents multiple circuits.

        If so, this object has .circuits and .pulse_lists attributes. Otherwise, this object
        represents a single circuit, and has .circuit and .pulse_list attributes.

        Returns:
            A boolean indicating whether this object represents multiple circuits.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        if not self.has_multiple_circuits():
            return (
                f"CompilerOutput({self.circuit!r}, {self.initial_logical_to_physical!r}, "
                f"{self.final_logical_to_physical!r}, {self.pulse_gate_circuit!r}, "
                f"{self.seq!r}, {self.jaqal_program!r}, {self.pulse_list!r})"
            )
        return (
            f"CompilerOutput({self.circuits!r}, {self.initial_logical_to_physicals!r}, "
            f"{self.final_logical_to_physicals!r}, {self.pulse_gate_circuits!r}, "
            f"{self.seq!r}, {self.jaqal_programs!r}, {self.pulse_lists!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CompilerOutput):
            return False

        if self.has_multiple_circuits() != other.has_multiple_circuits():
            return False
        elif self.has_multiple_circuits():
            return (
                self.circuits == other.circuits
                and self.initial_logical_to_physicals == other.initial_logical_to_physicals
                and self.final_logical_to_physicals == other.final_logical_to_physicals
                and self.pulse_gate_circuits == other.pulse_gate_circuits
                and self.jaqal_programs == other.jaqal_programs
                and self.pulse_lists == other.pulse_lists
                and self.seq == other.seq
            )

        return (
            self.circuit == other.circuit
            and self.initial_logical_to_physical == other.initial_logical_to_physical
            and self.final_logical_to_physical == other.final_logical_to_physical
            and self.pulse_gate_circuit == other.pulse_gate_circuit
            and self.jaqal_program == other.jaqal_program
            and self.pulse_list == other.pulse_list
            and self.seq == other.seq
        )


def read_json(json_dict: dict[str, str], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's compilation endpoints.

    Args:
        json_dict: A JSON dictionary matching the format returned by /compile endpoint.
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).

    Returns:
        A `CompilerOutput` object with the compiled circuit(s) and (if applicable to this target)
        corresponding pulse gate circuit(s).
    """
    compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])

    initial_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(json_dict["final_logical_to_physicals"]))
    )

    pulse_gate_circuits = pulse_sequences = None

    if "pulse_gate_circuits" in json_dict:
        pulse_gate_circuits = qss.deserialize_circuits(json_dict["pulse_gate_circuits"])

    if "pulses" in json_dict:
        try:
            pulse_sequences = gss.serialization.deserialize(json_dict["pulses"])
        except Exception:
            pulse_sequences = None

    if circuits_is_list:
        return CompilerOutput(
            compiled_circuits,
            initial_logical_to_physicals,
            final_logical_to_physicals,
            pulse_gate_circuits=pulse_gate_circuits,
            pulse_sequences=pulse_sequences,
        )
    return CompilerOutput(
        compiled_circuits[0],
        initial_logical_to_physicals[0],
        final_logical_to_physicals[0],
        pulse_gate_circuits=None if pulse_gate_circuits is None else pulse_gate_circuits[0],
        pulse_sequences=None if pulse_sequences is None else pulse_sequences[0],
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
        the returned object also stores the pulse sequence in the .seq attribute and the
        list(s) of cycles in the .pulse_list(s) attribute.
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
    pulse_lists = None

    if "state_jp" not in json_dict:
        warnings.warn(
            "This output only contains compiled circuits (using a default AQT gate set). To "
            "get back the corresponding pulse sequence, you must first upload your `qtrl` configs "
            "using `provider.aqt_upload_configs`."
        )
    elif not importlib.util.find_spec("qtrl"):
        warnings.warn(
            "This output only contains compiled circuits. The `qtrl` package must be installed in "
            "order to deserialize compiled pulse sequences."
        )
    else:  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs

        def _sequencer_from_state(state: dict[str, Any]) -> qtrl.sequencer.Sequence:
            seq = qtrl.sequencer.Sequence(n_elements=1)
            seq.__setstate__(state)
            seq.compile()
            return seq

        pulse_lists = gss.serialization.deserialize(json_dict["pulse_lists_jp"])
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

        pulse_lists = pulse_lists and [
            pulse_lists[i : i + num_eca_circuits]
            for i in range(0, len(pulse_lists), num_eca_circuits)
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
            pulse_lists=pulse_lists,
        )

    pulse_lists = pulse_lists[0] if pulse_lists is not None else None
    return CompilerOutput(
        compiled_circuits[0],
        initial_logical_to_physicals[0],
        final_logical_to_physicals[0],
        seq=seq,
        pulse_lists=pulse_lists,
    )


def read_json_qscout(
    json_dict: dict[str, str | list[str]],
    circuits_is_list: bool,
) -> CompilerOutput:
    """Reads out the returned JSON from Superstaq API's QSCOUT compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /qscout_compile endpoint.
        circuits_is_list: Bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False).

    Returns:
        A `CompilerOutput` object with the compiled circuit(s) and a list of
        jaqal programs in a string representation.
    """
    qiskit_circuits = json_dict["qiskit_circuits"]
    jaqal_programs = json_dict["jaqal_programs"]

    initial_logical_to_physicals_str = json_dict["initial_logical_to_physicals"]
    assert isinstance(initial_logical_to_physicals_str, str)
    initial_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(initial_logical_to_physicals_str))
    )

    final_logical_to_physicals_str = json_dict["final_logical_to_physicals"]
    assert isinstance(final_logical_to_physicals_str, str)
    final_logical_to_physicals: list[dict[int, int]] = list(
        map(dict, json.loads(final_logical_to_physicals_str))
    )

    assert isinstance(qiskit_circuits, str)
    assert isinstance(jaqal_programs, list)
    compiled_circuits = qss.serialization.deserialize_circuits(qiskit_circuits)

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
        jaqal_programs=jaqal_programs[0],
    )
