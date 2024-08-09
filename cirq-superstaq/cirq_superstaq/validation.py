# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Sequence

import cirq


def validate_cirq_circuits(circuits: object, require_measurements: bool = False) -> None:
    """Validates that the input is either a single `cirq.Circuit` or a list of `cirq.Circuit`
    instances.

    Args:
        circuits: The circuit(s) to run.
        require_measurements: An optional boolean flag to check if all of the circuits have
            measurements.

    Raises:
        ValueError: If the input is not a `cirq.Circuit` or a list of `cirq.Circuit` instances.
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
