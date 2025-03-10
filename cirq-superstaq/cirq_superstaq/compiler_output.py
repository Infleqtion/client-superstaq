from __future__ import annotations

import importlib.util
import json
import warnings
from typing import Any

import cirq
import general_superstaq as gss

import cirq_superstaq as css

try:
    import qtrl.sequence_utils.readout
except ModuleNotFoundError:
    pass


def active_qubit_indices(circuit: cirq.AbstractCircuit) -> list[int]:
    """Returns the indices of the non-idle qubits in a quantum circuit.

    Note:
        The "index" refers to the argument of a `LineQubit` (so e.g. `cirq.LineQubit(5)`
        has index 5 regardless of the total number of qubits in the circuit).

    Args:
        circuit: The input quantum circuit.

    Returns:
        A list of active qubit indicies.

    Raises:
        ValueError: If qubit indices are requested for non-line qubits.
    """

    all_qubits: set[cirq.Qid] = set()
    for op in circuit.all_operations():
        if not isinstance(op.gate, css.Barrier):
            all_qubits.update(op.qubits)

    qubit_indices: list[int] = []
    for q in sorted(all_qubits):
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise ValueError("Qubit indices can only be determined for line qubits.")
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
        A list of the measurement qubit indicies.

    Raises:
        ValueError: If qubit indices are requested for non-line qubits.
    """

    unrolled_circuit = cirq.unroll_circuit_op(circuit, deep=True, tags_to_check=None)

    measured_qubits: set[cirq.Qid] = set()
    for _, op in unrolled_circuit.findall_operations(cirq.is_measurement):
        measured_qubits.update(op.qubits)

    qubit_indices: set[int] = set()
    for q in measured_qubits:
        if not isinstance(q, (cirq.LineQubit, cirq.LineQid)):
            raise ValueError("Qubit indices can only be determined for line qubits")
        qubit_indices.add(int(q))

    return sorted(qubit_indices)


class CompilerOutput:
    """A class that arranges compiled circuit information."""

    def __init__(
        self,
        circuits: cirq.Circuit | list[cirq.Circuit] | list[list[cirq.Circuit]],
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
        pulse_gate_circuits: Any | None = None,
        seq: qtrl.sequencer.Sequence | None = None,
        jaqal_programs: list[str] | str | None = None,
    ) -> None:
        """Initializes the `CompilerOutput` attributes.

        Args:
            circuits: A list (of at most 2 dimensions) containing `cirq.Circuit` objects.
            initial_logical_to_physicals: Pre-compilation mapping of logical qubits to physical
                qubits.
            final_logical_to_physicals: Post-compilation mapping of logical qubits to physical
                qubits.
            pulse_gate_circuits: Pulse-gate `qiskit.QuantumCircuit` or list thereof specifying the
                pulse compilation.
            seq: A `qtrl` pulse sequence, if `qtrl` is available locally.
            jaqal_programs: The Jaqal program (resp. programs) as a string (resp. list of
                strings).
        """
        if isinstance(circuits, cirq.Circuit):
            self.circuit = circuits
            self.initial_logical_to_physical = initial_logical_to_physicals
            self.final_logical_to_physical = final_logical_to_physicals
            self.pulse_gate_circuit = pulse_gate_circuits
            self.jaqal_program = jaqal_programs
        else:
            self.circuits = circuits
            self.initial_logical_to_physicals = initial_logical_to_physicals
            self.final_logical_to_physicals = final_logical_to_physicals
            self.pulse_gate_circuits = pulse_gate_circuits
            self.jaqal_programs = jaqal_programs

        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Checks if this object has plural attributes (e.g. `.circuits`).

        Otherwise, the object represents a single circuit, and has singular attributes (`.circuit`).

        Returns:
            `True` if this object represents multiple circuits; `False` otherwise.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        if not self.has_multiple_circuits():
            return (
                f"CompilerOutput({self.circuit!r}, {self.initial_logical_to_physical!r}, "
                f"{self.final_logical_to_physical!r}, {self.pulse_gate_circuit!r}, "
                f"{self.seq!r}, {self.jaqal_program!r})"
            )
        return (
            f"CompilerOutput({self.circuits!r}, {self.initial_logical_to_physicals!r}, "
            f"{self.final_logical_to_physicals!r}, {self.pulse_gate_circuits!r}, "
            f"{self.seq!r}, {self.jaqal_programs!r})"
        )


def read_json(json_dict: dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
    """Reads out returned JSON from Superstaq API's IBMQ compilation endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by /ibmq_compile endpoint
        circuits_is_list: A bool flag that controls whether the returned object has a .circuits
            attribute (if `True`) or a .circuit attribute (`False`).

    Returns:
        A `CompilerOutput` object with the compiled circuit(s). If included in the server response,
        the returned object also stores the corresponding pulse gate circuit(s) in its
        .pulse_gate_circuit(s) attribute (provided qiskit-superstaq is available locally).
    """

    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
    initial_logical_to_physicals: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["initial_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )
    pulse_gate_circuits = None

    if "pulse_gate_circuits" in json_dict:
        pulse_gate_circuits = css.serialization.deserialize_qiskit_circuits(
            json_dict["pulse_gate_circuits"],
            circuits_is_list,
            pulse_durations=json_dict.get("pulse_durations"),
            pulse_start_times=json_dict.get("pulse_start_times"),
        )

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
    json_dict: dict[str, Any], circuits_is_list: bool, num_eca_circuits: int | None = None
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
        the returned object also stores the pulse sequence in the .seq attribute.
    """

    compiled_circuits: list[cirq.Circuit] | list[list[cirq.Circuit]]
    compiled_circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])

    initial_logical_to_physicals_list: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["initial_logical_to_physicals"]))
    )
    initial_logical_to_physicals: (
        list[dict[cirq.Qid, cirq.Qid]] | list[list[dict[cirq.Qid, cirq.Qid]]]
    ) = initial_logical_to_physicals_list

    final_logical_to_physicals_list: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )
    final_logical_to_physicals: (
        list[dict[cirq.Qid, cirq.Qid]] | list[list[dict[cirq.Qid, cirq.Qid]]]
    ) = final_logical_to_physicals_list

    seq = None

    if "state_jp" in json_dict:
        if not importlib.util.find_spec("qtrl"):
            warnings.warn(
                "This output only contains compiled circuits. The `qtrl` package must be installed "
                "in order to deserialize compiled pulse sequences."
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


def read_json_qscout(json_dict: dict[str, Any], circuits_is_list: bool) -> CompilerOutput:
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
    initial_logical_to_physicals: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["initial_logical_to_physicals"]))
    )
    final_logical_to_physicals: list[dict[cirq.Qid, cirq.Qid]] = list(
        map(dict, cirq.read_json(json_text=json_dict["final_logical_to_physicals"]))
    )

    if circuits_is_list:
        return CompilerOutput(
            circuits=compiled_circuits,
            initial_logical_to_physicals=initial_logical_to_physicals,
            final_logical_to_physicals=final_logical_to_physicals,
            jaqal_programs=json_dict["jaqal_programs"],
        )

    return CompilerOutput(
        circuits=compiled_circuits[0],
        initial_logical_to_physicals=initial_logical_to_physicals[0],
        final_logical_to_physicals=final_logical_to_physicals[0],
        jaqal_programs=json_dict["jaqal_programs"][0],
    )
