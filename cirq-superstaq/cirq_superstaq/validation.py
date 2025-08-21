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

from collections.abc import Sequence

import cirq

SUPPORTED_QID_TYPES = (
    cirq.LineQubit,
    cirq.LineQid,
    cirq.GridQubit,
    cirq.GridQid,
    cirq.NamedQubit,
    cirq.NamedQid,
)


def validate_qubit_types(circuits: cirq.Circuit | Sequence[cirq.Circuit]) -> None:
    """Verifies that `circuits` consists of valid (`cirq-core`) qubit types only.

    Args:
        circuits: The input circuit(s) to validate.

    Raises:
        TypeError: If an unsupported qubit type is found in `circuits`.
    """
    circuits_to_check = [circuits] if isinstance(circuits, cirq.Circuit) else circuits

    all_qubits_present: set[cirq.Qid] = set()
    for circuit in circuits_to_check:
        all_qubits_present.update(circuit.all_qubits())

    if not all(isinstance(q, SUPPORTED_QID_TYPES) for q in all_qubits_present):
        invalid_qubit_types = ", ".join(
            map(str, ({type(q) for q in all_qubits_present} - set(SUPPORTED_QID_TYPES)))
        )
        raise TypeError(
            f"Input circuit(s) contains unsupported qubit types: {invalid_qubit_types}. "
            "Valid qubit types are: `cirq.LineQubit`, `cirq.LineQid`, `cirq.GridQubit`, "
            "`cirq.GridQid`, `cirq.NamedQubit`, and `cirq.NamedQid`."
        )


def validate_cirq_circuits(circuits: object, require_measurements: bool = False) -> None:
    """Validates that the input is an acceptable `cirq-core` object for `cirq-superstaq`.

    In particular, this function verifies that `circuits` is either a single `cirq.Circuit`
    or a list of `cirq.Circuit` instances. Additionally, also validates that `circuits`
    contains supported qubit types only.

    Args:
        circuits: The circuit(s) to run.
        require_measurements: An optional boolean flag to check if all of the circuits have
            measurements.

    Raises:
        ValueError: If the input is not a `cirq.Circuit` or a list of `cirq.Circuit` instances.
        TypeError: If an unsupported qubit type is found in `circuits`.
    """
    if not (
        isinstance(circuits, cirq.Circuit)
        or (
            isinstance(circuits, Sequence)
            and all(isinstance(circuit, cirq.Circuit) for circuit in circuits)
        )
    ):
        raise ValueError(
            "Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
            "sequence of `cirq.Circuit` instances."
        )

    if require_measurements:
        circuit_list = [circuits] if isinstance(circuits, cirq.Circuit) else circuits
        for circuit in circuit_list:
            if isinstance(circuit, cirq.Circuit) and not circuit.has_measurements():
                # TODO: only raise if the run method actually requires samples (and not for e.g. a
                # statevector simulation)
                raise ValueError("Circuit has no measurements to sample.")

    validate_qubit_types(circuits)
