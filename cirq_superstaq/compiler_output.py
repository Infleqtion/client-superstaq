import importlib
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
    """Returns the indices of the non-idle qubits in a quantum circuit, where "index" refers to the
    argument of a LineQubit (so e.g. cirq.LineQubit(5) has index 5 regardless of the total number
    of qubits in the circuit)."""

    all_qubits: Set[cirq.Qid] = set()
    for op in circuit.all_operations():
        if not isinstance(op.gate, css.Barrier):
            all_qubits.update(op.qubits)

    qubit_indices: List[int] = []
    for q in sorted(all_qubits):
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise ValueError("Qubit indices can only be determined for line qubits")
        qubit_indices.append(int(q))

    return qubit_indices


def measured_qubit_indices(circuit: cirq.AbstractCircuit) -> List[int]:
    """Returns the indices of the measured qubits in a quantum circuit, where "index" refers to the
    argument of a LineQubit (so e.g. cirq.LineQubit(5) has index 5 regardless of the total number
    of qubits in the circuit)."""

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
    def __init__(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit], List[List[cirq.Circuit]]],
        pulse_sequences: Optional[Any] = None,
        seq: Optional["qtrl.sequencer.Sequence"] = None,
        jaqal_programs: Optional[Union[List[str], str]] = None,
        pulse_lists: Optional[Union[List[List[List[Any]]], List[List[List[List[Any]]]]]] = None,
    ) -> None:
        if isinstance(circuits, cirq.Circuit):
            self.circuit = circuits
            self.pulse_list = pulse_lists
            self.pulse_sequence = pulse_sequences
            self.jaqal_program = jaqal_programs
        else:
            self.circuits = circuits
            self.pulse_lists = pulse_lists
            self.pulse_sequences = pulse_sequences
            self.jaqal_programs = jaqal_programs

        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Returns True if this object represents multiple circuits.

        If so, this object has .circuits and .pulse_lists attributes. Otherwise, this object
        represents a single circuit, and has .circuit and .pulse_list attributes.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        if not self.has_multiple_circuits():
            return (
                f"CompilerOutput({self.circuit!r}, {self.pulse_sequence!r}, {self.seq!r}, "
                f"{self.jaqal_program!r}, {self.pulse_list!r})"
            )
        return (
            f"CompilerOutput({self.circuits!r}, {self.pulse_sequences!r}, {self.seq!r}, "
            f"{self.jaqal_programs!r}, {self.pulse_lists!r})"
        )


def read_json_ibmq(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from SuperstaQ API's IBMQ compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /ibmq_compile endpoint
        circuits_is_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a CompilerOutput object with the compiled circuit(s). If qiskit is available locally,
        the returned object also stores the pulse sequences in the .pulse_sequence(s) attribute.
    """
    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
    pulses = None

    if importlib.util.find_spec("qiskit"):
        import qiskit

        if "0.22" < qiskit.__version__ < "0.23":
            pulses = gss.serialization.deserialize(json_dict["pulses"])
        else:
            warnings.warn(
                "ibmq_compile requires Qiskit Terra version 0.20.* to deserialize compiled pulse "
                f"sequences (you have {qiskit.__version__})."
            )
    else:
        warnings.warn(
            "ibmq_compile requires Qiskit Terra version 0.20.* to deserialize compiled pulse "
            "sequences."
        )

    if circuits_is_list:
        return CompilerOutput(circuits=compiled_circuits, pulse_sequences=pulses)
    return CompilerOutput(circuits=compiled_circuits[0], pulse_sequences=pulses and pulses[0])


def read_json_aqt(
    json_dict: Dict[str, Any], circuits_is_list: bool, num_eca_circuits: int = 0
) -> CompilerOutput:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /aqt_compile endpoint
        circuits_is_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a CompilerOutput object with the compiled circuit(s). If qtrl is available locally,
        the returned object also stores the pulse sequence in the .seq attribute and the
        list(s) of cycles in the .pulse_list(s) attribute.
    """

    compiled_circuits: Union[List[cirq.Circuit], List[List[cirq.Circuit]]]
    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    seq = None
    pulse_lists = None

    if importlib.util.find_spec(
        "qtrl"
    ):  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs

        def _sequencer_from_state(state: Dict[str, Any]) -> "qtrl.sequencer.Sequence":
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
    else:
        warnings.warn(
            "Your sequence for this output is None. Please make sure you have the qtrl package "
            "installed in order to deserialize compiled pulse sequences."
        )

    if num_eca_circuits:
        compiled_circuits = [
            compiled_circuits[i : i + num_eca_circuits]
            for i in range(0, len(compiled_circuits), num_eca_circuits)
        ]

        pulse_lists = pulse_lists and [
            pulse_lists[i : i + num_eca_circuits]
            for i in range(0, len(pulse_lists), num_eca_circuits)
        ]

    if circuits_is_list:
        return CompilerOutput(circuits=compiled_circuits, seq=seq, pulse_lists=pulse_lists)

    pulse_lists = pulse_lists[0] if pulse_lists is not None else None
    return CompilerOutput(circuits=compiled_circuits[0], seq=seq, pulse_lists=pulse_lists)


def read_json_qscout(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from SuperstaQ API's QSCOUT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /qscout_compile endpoint
        circuits_is_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a CompilerOutput object with the compiled circuit(s) and a list jaqal programs
        represented as strings
    """

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    if circuits_is_list:
        return CompilerOutput(
            circuits=compiled_circuits, jaqal_programs=json_dict["jaqal_programs"]
        )

    return CompilerOutput(
        circuits=compiled_circuits[0], jaqal_programs=json_dict["jaqal_programs"][0]
    )


def read_json_only_circuits(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from SuperstaQ API's CQ compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /cq_compile endpoint
        circuits_is_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a CompilerOutput object with the compiled circuit(s)
    """

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    if circuits_is_list:
        return CompilerOutput(circuits=compiled_circuits)

    return CompilerOutput(circuits=compiled_circuits[0])


def read_json_neutral_atom(json_dict: Dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    try:
        pulses = gss.serialization.deserialize(json_dict["pulses"])
    except ModuleNotFoundError as e:
        warnings.warn(
            f"neutral_atom_compile requires {e.name} to deserialize compiled pulse sequences."
        )
        pulses = None

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    if circuits_is_list:
        return CompilerOutput(circuits=compiled_circuits, pulse_sequences=pulses)

    return CompilerOutput(circuits=compiled_circuits[0], pulse_sequences=pulses and pulses[0])
