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
"""Methods to count resources in a circuit."""

from __future__ import annotations

import functools
from collections.abc import Callable

import cirq

import cirq_superstaq as css


def num_category_ops(
    category_classifier: Callable[[cirq.Operation], bool], circuit: cirq.Circuit
) -> int:
    """Computes the number of operations in a circuit that belong to a category.

    Args:
        category_classifier: A `Callable` that returns true if the given operation has
            certain features.
        circuit: A Cirq circuit.

    Returns:
        A number representing how many operations in the given circuit
        match the category classifier's conditions.
    """
    return sum(category_classifier(op) for op in circuit.all_operations())


def num_single_qubit_gates(
    circuit: cirq.Circuit,
) -> int:
    """Get number of single qubit gates in a circuit.

    Args:
        circuit: A Cirq circuit.

    Returns:
        Number of single qubit gates.
    """
    return num_category_ops(lambda op: len(op.qubits) == 1, circuit)


def num_two_qubit_gates(circuit: cirq.Circuit) -> int:
    """Get number of two qubit gates in a circuit.

    Args:
        circuit: A Cirq circuit.

    Results:
        Number of two qubit gates.
    """
    return num_category_ops(lambda op: len(op.qubits) == 2, circuit)


def num_phased_xpow_subgates(
    circuit: cirq.Circuit,
) -> int:
    """Get number of non-diagonal single qubit gates in a circuit.

    Args:
        circuit: A Cirq circuit.

    Results:
        Number of non-diagonal single qubit gates.
    """

    def _contains_x_pow_subgate(op: cirq.Operation) -> bool:
        return len(op.qubits) == 1 and not cirq.is_diagonal(cirq.unitary(op))

    return num_category_ops(_contains_x_pow_subgate, circuit)


def _is_global_op(op: cirq.Operation, circuit: cirq.Circuit) -> bool:
    """Check if an operation in a circuit is applied to all qubits.

    Args:
        op: Operation to check.
        circuit: A Cirq circuit.

    Returns:
        A boolean representing whether the operation is applied to all qubits.
    """
    return isinstance(
        op.gate, (cirq.ParallelGate, cirq.InterchangeableQubitsGate)
    ) and cirq.num_qubits(op) == cirq.num_qubits(circuit)


def num_global_ops(circuit: cirq.Circuit) -> int:
    """Get number of global operations in a circuit.

    Args:
        circuit: A Cirq circuit.

    Returns:
        Number of global operations.
    """
    return num_category_ops(functools.partial(_is_global_op, circuit=circuit), circuit)


def total_global_rgate_pi_time(
    circuit: cirq.Circuit,
) -> float:
    """Get sum of rotation exponents applied by global parallel `RGate` gates in a circuit.

    Args:
        circuit: A Cirq circuit.

    Returns:
        Float with value being the sum of rotation exponents.
    """
    return sum(
        abs(css.evaluation.paramval_to_float(op.gate.exponent))
        for op in circuit.all_operations()
        if isinstance(op.gate, css.ParallelRGate) and _is_global_op(op, circuit)
    )
