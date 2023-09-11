from __future__ import annotations

import importlib.util
import json
import warnings
from typing import Any, Dict, List, Optional, Set, Union

import cirq
import general_superstaq as gss

import cirq_superstaq as css

try:
    import qtrl.sequence_utils.readout
except ModuleNotFoundError:
    pass


def active_qubit_indices(circuit: cirq.AbstractCircuit) -> List[int]:
    """Returns the indices of the non-idle qubits in a quantum circuit.

    Note:
        The "index" refers to the argument of a LineQubit (so e.g. `cirq.LineQubit(5)`
        has index 5 regardless of the total number of qubits in the circuit.

    Args:
        circuit: The input quantum circuit.

    Returns:
        A list of active qubit indicies.

    Raises:
        ValueError: If qubit indices are requested for non-line qubits.
    """

    all_qubits: Set[cirq.Qid] = set()
    for op in circuit.all_operations():
        if not isinstance(op.gate, css.Barrier):
            all_qubits.update(op.qubits)

    qubit_indices: List[int] = []
    for q in sorted(all_qubits):
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise ValueError("Qubit indices can only be determined for line qubits.")
        qubit_indices.append(int(q))

    return qubit_indices


def measured_qubit_indices(circuit: cirq.AbstractCircuit) -> List[int]:
    """Returns the indices of the measured qubits in a quantum circuit.

    Note:
        The "index" refers to the argument of a `cirq.LineQubit` (so e.g. `cirq.LineQubit(5)`
        has index 5 regardless of the total number of qubits in the circuit).

    Args:
        circuit: The input quantum circuit.

    Returns:
        A list of the measurement qubit indicies.

    Raises:
        ValueError: If qubit indices are requested for non-line qubits.
    """

    unrolled_circuit = cirq.unroll_circuit_op(circuit, deep=True, tags_to_check=None)

    measured_qubits: Set[cirq.Qid] = set()
    for _, op in unrolled_circuit.findall_operations(cirq.is_measurement):
        measured_qubits.update(op.qubits)

    qubit_indices: Set[int] = set()
    for q in measured_qubits:
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise ValueError("Qubit indices can only be determined for line qubits")
        qubit_indices.add(int(q))

    return sorted(qubit_indices)


class CompilerOutput:
    """A class that arranges compiled circuit information."""

    def __init__(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit], List[List[cirq.Circuit]]],
        final_logical_to_physicals: Union[
            Dict[cirq.Qid, cirq.Qid],
            List[Dict[cirq.Qid, cirq.Qid]],
            List[List[Dict[cirq.Qid, cirq.Qid]]],
        ],
        pulse_gate_circuits: Optional[Any] = None,
        pulse_sequences: Optional[Any] = None,
        seq: Optional[qtrl.sequencer.Sequence] = None,
        jaqal_programs: Optional[Union[List[str], str]] = None,
        pulse_lists: Optional[Union[List[List[List[Any]]], List[List[List[List[Any]]]]]] = None,
    ) -> None:
        """Initializes the `CompilerOutput` attributes.

        Args:
            circuits: A list (of at most 2 dimensions) containing `cirq.Circuit` objects.
            final_logical_to_physicals: Post-compilation mapping of logical qubits to physical
                qubits.
            pulse_gate_circuits: Pulse-gate `qiskit.QuantumCircuit` or list thereof specifying the
                pulse compilation.
            pulse_sequences: The qiskit pulse schedules for the compiled circuit(s).
            seq: A `qtrl` pulse sequence, if `qtrl` is available locally.
            jaqal_programs: The Jaqal program (resp. programs) as a string (resp. list of
                strings).
            pulse_lists: Optional list of pulse cycles if `qtrl` is available locally.
        """
        if isinstance(circuits, cirq.Circuit):
            self.circuit = circuits
            self.final_logical_to_physical = final_logical_to_physicals
            self.pulse_list = pulse_lists
            self.pulse_gate_circuit = pulse_gate_circuits
            self.pulse_sequence = pulse_sequences
            self.jaqal_program = jaqal_programs
        else:
            self.circuits = circuits
            self.final_logical_to_physicals = final_logical_to_physicals
            self.pulse_lists = pulse_lists
            self.pulse_gate_circuits = pulse_gate_circuits
            self.pulse_sequences = pulse_sequences
            self.jaqal_programs = jaqal_programs

        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Checks if an object has .circuits and .pulse_lists attributes.

        Otherwise, the object represents a single circuit, and has .circuit
        and .pulse_list attributes.

        Returns:
            `True` if this object represents multiple circuits; `False` otherwise.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        if not self.has_multiple_circuits():
            return (
                f"CompilerOutput({self.circuit!r}, {self.final_logical_to_physical!r}, "
                f"{self.pulse_gate_circuit!r}, {self.pulse_sequence!r}, {self.seq!r}, "
                f"{self.jaqal_program!r}, {self.pulse_list!r})"
            )
        return (
            f"CompilerOutput({self.circuits!r}, {self.final_logical_to_physicals!r}, "
            f"{self.pulse_gate_circuits!r}, {self.pulse_sequences!r}, {self.seq!r}, "
            f"{self.jaqal_programs!r}, {self.pulse_lists!r})"
        )


def _deserialize_qiskit_circuits(
    serialized_qiskit_circuits: str, circuits_is_list: bool
) -> Optional[List[Any]]:
    """Deserializes `qiskit.QuantumCircuit` objects, if possible; otherwise warns the user.

    Args:
        serialized_qiskit_circuits: Qiskit circuits serialized via `qss.serialize_circuits()`.
        circuits_is_list: Whether to refer to "circuits" (plural) or "circuit" (singular) in warning
            messages.

    Returns:
        A list of deserialized `qiskit.QuantumCircuit` objects, or None if the provided circuits
        could not be deserialized.
    """
    if importlib.util.find_spec("qiskit_superstaq"):
        import qiskit
        import qiskit_superstaq as qss

        try:
            return qss.deserialize_circuits(serialized_qiskit_circuits)
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
                "attribute of this output."
            )

    else:
        s = "s" if circuits_is_list else ""
        warnings.warn(
            "qiskit-superstaq is required to deserialize compiled pulse gate circuits. You can "
            "install it with `pip install qiskit-superstaq`.\n\n"
            f"You can still access your compiled circuit{s} using the .circuit{s} attribute of "
            "this output."
        )

    return None


def read_json(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's IBMQ compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /ibmq_compile endpoint
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).

    Returns:
        A `CompilerOutput` object with the compiled circuit(s). If included in the server response,
        the returned object also stores the corresponding pulse gate circuit(s) in its
        .pulse_gate_circuit(s) attribute, and pulse sequence(s) in its .pulse_sequences(s) attribute
        (provided qiskit-superstaq is available locally).
    """

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
    final_logical_to_physicals: List[Dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )
    pulse_gate_circuits = pulses = None

    if "pulse_gate_circuits" in json_dict:
        pulse_gate_circuits = _deserialize_qiskit_circuits(
            json_dict["pulse_gate_circuits"], circuits_is_list
        )

    if "pulses" in json_dict:
        if importlib.util.find_spec("qiskit") and importlib.util.find_spec("qiskit.qpy"):
            import qiskit

            try:
                pulses = gss.serialization.deserialize(json_dict["pulses"])
            except Exception as e:
                s = "s" if circuits_is_list else ""
                if qiskit.__version__ < "0.24":
                    warnings.warn(
                        f"Your compiled pulse sequence{s} could not be deserialized, likely "
                        f"because your Qiskit Terra installation (version {qiskit.__version__}) is "
                        "out of date. Please try again after installing a more recent version.\n\n"
                        f"You can still access your compiled circuit{s} using the .circuit{s} "
                        "attribute of this output."
                    )
                else:
                    warnings.warn(
                        f"Your compiled pulse sequence{s} could not be deserialized. Please let "
                        "us know at superstaq@infleqtion.com, or file a report at "
                        "https://github.com/Infleqtion/client-superstaq/issues containing "
                        "the following information (as well as any other relevant context):\n\n"
                        f"cirq-superstaq version: {css.__version__}\n"
                        f"qiskit-terra version: {qiskit.__version__}\n"
                        f"error: {e!r}\n\n"
                        f"You can still access your compiled circuit{s} using the .circuit{s} "
                        "attribute of this output."
                    )
        else:
            s = "s" if circuits_is_list else ""
            warnings.warn(
                f"Qiskit Terra is required to deserialize compiled pulse sequence{s}. You can "
                f"still access your compiled circuit{s} using the .circuit{s} attribute of this "
                "output."
            )

    if circuits_is_list:
        return CompilerOutput(
            compiled_circuits,
            final_logical_to_physicals,
            pulse_gate_circuits=pulse_gate_circuits,
            pulse_sequences=pulses,
        )
    return CompilerOutput(
        compiled_circuits[0],
        final_logical_to_physicals[0],
        pulse_gate_circuits=None if pulse_gate_circuits is None else pulse_gate_circuits[0],
        pulse_sequences=None if pulses is None else pulses[0],
    )


def read_json_aqt(
    json_dict: Dict[str, Any], circuits_is_list: bool, num_eca_circuits: Optional[int] = None
) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's AQT compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by aqt_compile endpoint.
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).
        num_eca_circuits: Number of logically equivalent random circuits to generate for each
            input circuit.

    Returns:
        A `CompilerOutput` object with the compiled circuit(s). If `qtrl` is available locally,
        the returned object also stores the pulse sequence in the .seq attribute and the
        list(s) of cycles in the .pulse_list(s) attribute.
    """

    compiled_circuits: Union[List[cirq.Circuit], List[List[cirq.Circuit]]]
    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    final_logical_to_physicals_list: List[Dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )
    final_logical_to_physicals: Union[
        List[Dict[cirq.Qid, cirq.Qid]], List[List[Dict[cirq.Qid, cirq.Qid]]]
    ] = final_logical_to_physicals_list

    seq = None
    pulse_lists = None

    if "state_jp" not in json_dict:
        warnings.warn(
            "This output only contains compiled circuits (using a default AQT gate set). To "
            "get back the corresponding pulse sequence, you must first upload your `qtrl` configs "
            "using `service.aqt_upload_configs`."
        )
    elif not importlib.util.find_spec("qtrl"):
        warnings.warn(
            "This output only contains compiled circuits. The `qtrl` package must be installed in "
            "order to deserialize compiled pulse sequences."
        )
    else:  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs

        def _sequencer_from_state(state: Dict[str, Any]) -> qtrl.sequencer.Sequence:
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
        final_logical_to_physicals = [
            final_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(final_logical_to_physicals_list), num_eca_circuits)
        ]
        pulse_lists = pulse_lists and [
            pulse_lists[i : i + num_eca_circuits]
            for i in range(0, len(pulse_lists), num_eca_circuits)
        ]

    if circuits_is_list:
        return CompilerOutput(
            compiled_circuits, final_logical_to_physicals, seq=seq, pulse_lists=pulse_lists
        )

    pulse_lists = pulse_lists[0] if pulse_lists is not None else None
    return CompilerOutput(
        compiled_circuits[0], final_logical_to_physicals[0], seq=seq, pulse_lists=pulse_lists
    )


def read_json_qscout(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's QSCOUT compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /qscout_compile endpoint.
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).

    Returns:
        A `CompilerOutput` object with the compiled circuit(s) and a list of jaqal programs
        represented as strings.
    """

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
    final_logical_to_physicals: List[Dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )

    if circuits_is_list:
        return CompilerOutput(
            circuits=compiled_circuits,
            final_logical_to_physicals=final_logical_to_physicals,
            jaqal_programs=json_dict["jaqal_programs"],
        )

    return CompilerOutput(
        circuits=compiled_circuits[0],
        final_logical_to_physicals=final_logical_to_physicals[0],
        jaqal_programs=json_dict["jaqal_programs"][0],
    )
