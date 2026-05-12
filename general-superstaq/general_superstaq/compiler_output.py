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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import general_superstaq as gss

try:
    import qtrl.sequence_utils.readout
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    from typing_extensions import Self

C = TypeVar("C")
Q = TypeVar("Q")


class BaseCompilerOutput(Generic[C, Q]):  # noqa: PLW1641
    """A base class that stores the results of compiled circuits."""

    def __init__(
        self,
        circuits: (C | list[C] | list[list[C]]),
        initial_logical_to_physicals: (dict[Q, Q] | list[dict[Q, Q]] | list[list[dict[Q, Q]]]),
        final_logical_to_physicals: (dict[Q, Q] | list[dict[Q, Q]] | list[list[dict[Q, Q]]]),
        pulse_gate_circuits: Any | None = None,
        seq: Any | None = None,
        jaqal_programs: list[str] | None = None,
    ) -> None:
        """Constructs a `BaseCompilerOutput` object.

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
        if isinstance(circuits, list):
            self.circuits = circuits
            self.initial_logical_to_physicals = initial_logical_to_physicals
            self.final_logical_to_physicals = final_logical_to_physicals
            self.pulse_gate_circuits = pulse_gate_circuits
        else:
            self.circuit = circuits
            self.initial_logical_to_physical = initial_logical_to_physicals
            self.final_logical_to_physical = final_logical_to_physicals
            self.pulse_gate_circuit = pulse_gate_circuits

        self.jaqal_programs = jaqal_programs
        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Checks if this object has plural attributes (e.g. `.circuits`).

        Otherwise, the object represents a single circuit, and has singular attributes (`.circuit`).

        Returns:
            A boolean indicating whether this object represents multiple circuits.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if not self.has_multiple_circuits():
            return (
                f"{class_name}({self.circuit!r}, {self.initial_logical_to_physical!r}, "
                f"{self.final_logical_to_physical!r}, {self.pulse_gate_circuit!r}, "
                f"{self.seq!r}, {self.jaqal_programs!r})"
            )
        return (
            f"{class_name}({self.circuits!r}, {self.initial_logical_to_physicals!r}, "
            f"{self.final_logical_to_physicals!r}, {self.pulse_gate_circuits!r}, "
            f"{self.seq!r}, {self.jaqal_programs!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseCompilerOutput):
            return False

        if self.has_multiple_circuits() != other.has_multiple_circuits():
            return False
        if self.has_multiple_circuits():
            return (
                self.circuits == other.circuits
                and self.initial_logical_to_physicals == other.initial_logical_to_physicals
                and self.final_logical_to_physicals == other.final_logical_to_physicals
                and self.pulse_gate_circuits == other.pulse_gate_circuits
                and self.jaqal_programs == other.jaqal_programs
                and self.seq == other.seq
            )

        return (
            self.circuit == other.circuit
            and self.initial_logical_to_physical == other.initial_logical_to_physical
            and self.final_logical_to_physical == other.final_logical_to_physical
            and self.pulse_gate_circuit == other.pulse_gate_circuit
            and self.jaqal_programs == other.jaqal_programs
            and self.seq == other.seq
        )

    @property
    def jaqal_program(self) -> str | None:
        """Jaqal program(s) as a single string.

        For multi-circuit compilation the string will contain subcircuits.
        """
        if not self.jaqal_programs:
            return None

        return _jaqal_programs_to_subcircuits(self.jaqal_programs)

    @classmethod
    def _read_json(
        cls,
        deserialized_circuits: list[C],
        initial_logical_to_physicals: list[dict[Q, Q]],
        final_logical_to_physicals: list[dict[Q, Q]],
        pulse_gate_circuits: list[object] | None,
        circuits_is_list: bool,
    ) -> Self:
        """Reads out returned JSON from Superstaq API's IBMQ compilation endpoint.

        Args:
            json_dict: A JSON dictionary matching the format returned by /compile endpoint
            circuits_is_list: A bool flag that controls whether the returned object has a .circuits
                attribute (if `True`) or a .circuit attribute (`False`).

        Returns:
            A `CompilerOutput` object with the compiled circuit(s). If included in the server
            response, the returned object also stores the corresponding pulse gate circuit(s) in its
            .pulse_gate_circuit(s) attribute (provided `qiskit-superstaq` is available locally).
        """
        if circuits_is_list:
            return cls(
                deserialized_circuits,
                initial_logical_to_physicals,
                final_logical_to_physicals,
                pulse_gate_circuits=pulse_gate_circuits,
            )
        return cls(
            deserialized_circuits[0],
            initial_logical_to_physicals[0],
            final_logical_to_physicals[0],
            pulse_gate_circuits=None if pulse_gate_circuits is None else pulse_gate_circuits[0],
        )

    @classmethod
    def _read_json_jaqal(
        cls,
        json_dict: dict[str, Any],
        num_eca_circuits: int | None = None,
    ) -> Self:
        """Reads out the returned JSON from Superstaq API's Jaqal compilation endpoint.

        Args:
            json_dict: A JSON dictionary matching the format returned by `/compile` endpoint.
            num_eca_circuits: Number of logically equivalent random circuits to generate for each
                input circuit.

        Returns:
            A `CompilerOutput` object with the compiled Jaqal program(s).
        """
        compiled_circuits = json.loads(json_dict["jaqal_strs"])

        initial_logical_to_physicals_list: list[dict[Q, Q]] = list(
            map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
        )
        initial_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            initial_logical_to_physicals_list
        )

        final_logical_to_physicals_list: list[dict[Q, Q]] = list(
            map(dict, json.loads(json_dict["final_logical_to_physicals"]))
        )
        final_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            final_logical_to_physicals_list
        )

        jaqal_programs: list[str] = json_dict.get("jaqal_programs", compiled_circuits)
        if num_eca_circuits is not None:
            compiled_circuits, initial_logical_to_physicals, final_logical_to_physicals = (
                cls._format_eca_content(
                    compiled_circuits,
                    initial_logical_to_physicals_list,
                    final_logical_to_physicals_list,
                    num_eca_circuits,
                )
            )
            jaqal_programs = [
                _jaqal_programs_to_subcircuits(jaqal_programs[i : i + num_eca_circuits])
                for i in range(0, len(jaqal_programs), num_eca_circuits)
            ]

        return cls(
            circuits=compiled_circuits,
            initial_logical_to_physicals=initial_logical_to_physicals,
            final_logical_to_physicals=final_logical_to_physicals,
            jaqal_programs=jaqal_programs,
        )

    @classmethod
    def _read_json_aqt(
        cls,
        deserialized_circuits: list[C],
        initial_logical_to_physicals_list: list[dict[Q, Q]],
        final_logical_to_physicals_list: list[dict[Q, Q]],
        json_dict: dict[str, Any],
        circuits_is_list: bool,
        num_eca_circuits: int | None = None,
    ) -> Self:
        """Reads out returned JSON from Superstaq API's AQT compilation endpoint.

        Args:
            json_dict: A JSON dictionary matching the format returned by /aqt_compile endpoint.
            circuits_is_list: A bool flag that controls whether the returned object has a .circuits
                attribute (if `True`) or a .circuit attribute (`False`).
            num_eca_circuits: Number of logically equivalent random circuits to generate for each
                input circuit.

        Returns:
            A `CompilerOutput` object with the compiled circuit(s). If `qtrl` is available locally,
            the returned object also stores the pulse sequence in the .seq attribute.
        """
        compiled_circuits: list[C] | list[list[C]] = deserialized_circuits
        initial_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            initial_logical_to_physicals_list
        )

        final_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            final_logical_to_physicals_list
        )

        seq = None
        if "state_jp" in json_dict:
            if not importlib.util.find_spec("qtrl"):
                warnings.warn(
                    "This output only contains compiled circuits. The `qtrl` package must be "
                    "installed in order to deserialize compiled pulse sequences.",
                    stacklevel=2,
                )
            else:  # pragma: no cover, b/c qtrl is not open source so it is not in the reqs

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
                            readout_seq, readout_qubits, n_readouts=len(deserialized_circuits)
                        )

                    state["_readout"] = readout_seq

                seq = _sequencer_from_state(state)

        if num_eca_circuits is not None:
            compiled_circuits, initial_logical_to_physicals, final_logical_to_physicals = (
                cls._format_eca_content(
                    deserialized_circuits,
                    initial_logical_to_physicals_list,
                    final_logical_to_physicals_list,
                    num_eca_circuits,
                )
            )

        if circuits_is_list:
            return cls(
                compiled_circuits,
                initial_logical_to_physicals,
                final_logical_to_physicals,
                seq=seq,
            )

        return cls(
            compiled_circuits[0],
            initial_logical_to_physicals[0],
            final_logical_to_physicals[0],
            seq=seq,
        )

    @classmethod
    def _read_json_qscout(
        cls,
        deserialized_circuits: list[C],
        initial_logical_to_physicals_list: list[dict[Q, Q]],
        final_logical_to_physicals_list: list[dict[Q, Q]],
        jaqal_programs: list[str],
        circuits_is_list: bool,
        num_eca_circuits: int | None = None,
    ) -> Self:
        """Reads out returned JSON from Superstaq API's QSCOUT compilation endpoint.

        Args:
            json_dict: A JSON dictionary matching the format returned by /qscout_compile endpoint.
            circuits_is_list: A bool flag that controls whether the returned object has a .circuits
                attribute (if `True`) or a .circuit attribute (`False`).
            num_eca_circuits: Optional number of logically equivalent random circuits to generate
                for each input circuit.

        Returns:
            A `CompilerOutput` object with the compiled circuit(s) and a list of jaqal programs
            represented as strings.
        """
        compiled_circuits: list[C] | list[list[C]] = deserialized_circuits
        initial_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            initial_logical_to_physicals_list
        )
        final_logical_to_physicals: list[dict[Q, Q]] | list[list[dict[Q, Q]]] = (
            final_logical_to_physicals_list
        )

        if num_eca_circuits is not None:
            compiled_circuits, initial_logical_to_physicals, final_logical_to_physicals = (
                cls._format_eca_content(
                    deserialized_circuits,
                    initial_logical_to_physicals_list,
                    final_logical_to_physicals_list,
                    num_eca_circuits,
                )
            )
            jaqal_programs = [
                _jaqal_programs_to_subcircuits(jaqal_programs[i : i + num_eca_circuits])
                for i in range(0, len(jaqal_programs), num_eca_circuits)
            ]

        if circuits_is_list:
            return cls(
                compiled_circuits,
                initial_logical_to_physicals,
                final_logical_to_physicals,
                jaqal_programs=jaqal_programs,
            )

        return cls(
            compiled_circuits[0],
            initial_logical_to_physicals[0],
            final_logical_to_physicals[0],
            jaqal_programs=jaqal_programs,
        )

    @staticmethod
    def _get_deserialized_content(
        json_dict: dict[str, Any],
        _circuits_is_list: bool,
        _api_version: str = gss.API_VERSION,
    ) -> tuple[list[C], list[object] | None, list[dict[Q, Q]], list[dict[Q, Q]]]:
        compiled_circuits: list[C] = json.loads(json_dict["qasm_strs"])
        pulse_gate_circuits = None
        initial_logical_to_physicals_list: list[dict[Q, Q]] = list(
            map(dict, json.loads(json_dict["initial_logical_to_physicals"]))
        )
        final_logical_to_physicals_list: list[dict[Q, Q]] = list(
            map(dict, json.loads(json_dict["final_logical_to_physicals"]))
        )
        return (
            compiled_circuits,
            pulse_gate_circuits,
            initial_logical_to_physicals_list,
            final_logical_to_physicals_list,
        )

    @staticmethod
    def _format_eca_content(
        deserialized_circuits: list[C],
        initial_logical_to_physicals_list: list[dict[Q, Q]],
        final_logical_to_physicals_list: list[dict[Q, Q]],
        num_eca_circuits: int,
    ) -> tuple[list[list[C]], list[list[dict[Q, Q]]], list[list[dict[Q, Q]]]]:
        compiled_circuits = [
            deserialized_circuits[i : i + num_eca_circuits]
            for i in range(0, len(deserialized_circuits), num_eca_circuits)
        ]
        initial_logical_to_physicals = [
            initial_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(initial_logical_to_physicals_list), num_eca_circuits)
        ]
        final_logical_to_physicals = [
            final_logical_to_physicals_list[i : i + num_eca_circuits]
            for i in range(0, len(final_logical_to_physicals_list), num_eca_circuits)
        ]
        return compiled_circuits, initial_logical_to_physicals, final_logical_to_physicals

    @classmethod
    def _generate_compiler_output(
        cls,
        json_dict: dict[str, Any],
        *,
        circuits_is_list: bool,
        parser: str = "std",
        num_eca_circuits: int | None = None,
        api_version: str = gss.API_VERSION,
    ) -> Self:
        (
            compiled_circuits,
            pulse_gate_circuits,
            initial_logical_to_physicals_list,
            final_logical_to_physicals_list,
        ) = cls._get_deserialized_content(json_dict, circuits_is_list, api_version)

        if parser == "qscout":
            jaqal_programs: list[str] = json_dict["jaqal_programs"]
            return cls._read_json_qscout(
                compiled_circuits,
                initial_logical_to_physicals_list,
                final_logical_to_physicals_list,
                jaqal_programs,
                circuits_is_list,
                num_eca_circuits,
            )
        if parser == "aqt":
            return cls._read_json_aqt(
                compiled_circuits,
                initial_logical_to_physicals_list,
                final_logical_to_physicals_list,
                json_dict,
                circuits_is_list,
                num_eca_circuits,
            )
        return cls._read_json(
            compiled_circuits,
            initial_logical_to_physicals_list,
            final_logical_to_physicals_list,
            pulse_gate_circuits,
            circuits_is_list,
        )


class CompilerOutput(BaseCompilerOutput[str, int]):
    """A class that arranges compiled circuit information."""

    def __init__(
        self,
        circuits: str | list[str] | list[list[str]],
        initial_logical_to_physicals: dict[int, int]
        | list[dict[int, int]]
        | list[list[dict[int, int]]],
        final_logical_to_physicals: dict[int, int]
        | list[dict[int, int]]
        | list[list[dict[int, int]]],
        pulse_gate_circuits: Any | None = None,
        seq: Any | None = None,
        jaqal_programs: list[str] | None = None,
    ) -> None:
        """Constructs a `CompilerOutput` object for string-based `circuits`.

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


def _jaqal_programs_to_subcircuits(jaqal_programs: Sequence[str]) -> str:
    separator = "prepare_all"
    subcircuits = [jaqal_programs[0]]
    subcircuits += [jaqal_program.partition(separator)[2] for jaqal_program in jaqal_programs[1:]]
    return f"\n{separator}".join(subcircuits)
