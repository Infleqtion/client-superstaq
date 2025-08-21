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

# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import cirq
import numpy as np

import cirq_superstaq as css


def test_resource_counters() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.Z(qubits[2]),
        css.ParallelRGate(np.pi / 2, np.pi / 2, len(qubits)).on(*qubits),
        cirq.ISWAP(qubits[0], qubits[1]),
        cirq.CZ(qubits[1], qubits[2]),
        cirq.TOFFOLI(qubits[0], qubits[1], qubits[2]),
    )

    assert css.resource_counters.num_single_qubit_gates(circuit) == 3
    assert css.resource_counters.num_two_qubit_gates(circuit) == 2
    assert css.resource_counters.num_phased_xpow_subgates(circuit) == 2
    assert css.resource_counters.num_global_ops(circuit) == 2
    assert np.isclose(css.resource_counters.total_global_rgate_pi_time(circuit), 0.5)
