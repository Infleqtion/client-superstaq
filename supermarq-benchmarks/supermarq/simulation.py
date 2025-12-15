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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cirq


def get_ideal_counts(circuit: cirq.Circuit) -> dict[str, float]:
    """Noiseless statevector simulation.

    Note that the qubits in the returned bitstrings are in big-endian order.
    For example, for a circuit defined on qubits
    .. code::

        q0 ------
        q1 ------
        q2 ------

    the bitstrings are written as `q0q1q2`.

    Args:
        circuit: Input `cirq.Circuit` to be simulated.

    Returns:
        A dictionary with bitstring and probability as the key, value pairs.
    """
    ideal_counts = {}
    for i, amplitude in enumerate(circuit.final_state_vector(ignore_terminal_measurements=True)):
        bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
        probability = np.abs(amplitude) ** 2
        ideal_counts[bitstring] = probability
    return ideal_counts
