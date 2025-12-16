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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import cirq
import general_superstaq as gss
import networkx as nx
import numpy as np

import cirq_superstaq as css

if TYPE_CHECKING:
    import numpy.typing as npt


def paramval_to_float(param: cirq.TParamVal) -> float:
    """Converts a value of type `cirq.TParamVal` (`float | sympy.Expr`) to float.

    Args:
        param: The `cirq.TParamVal` to convert.

    Returns:
        A float representing the converted param.

    Raises:
        gss.SuperstaqException: If input parameter is not a type
            capable of being converted to float.
    """
    try:
        return float(param)
    except TypeError:
        raise gss.SuperstaqException("We don't support parametrized circuits yet.")


def is_known_diagonal(gate: cirq.Gate | None, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Checks if gate is represented by a diagonal matrix.

    If the given gate is a controlled gate, it checks if the uncontrolled version of the gate is
    diagonal.

    Args:
        gate: Gate to be checked.
        rtol: Relative tolerance when checking.
        atol: Absolute tolerance when checking.

    Returns:
        A boolean value representing whether `gate` is diagonal or not.
    """
    if isinstance(
        gate,
        (
            cirq.IdentityGate,
            cirq.DiagonalGate,
            cirq.ZPowGate,
            cirq.CZPowGate,
            cirq.CCZPowGate,
            cirq.ZZPowGate,
            css.StrippedCZGate,
        ),
    ):
        return True

    if isinstance(gate, cirq.ControlledGate):
        return is_known_diagonal(gate.sub_gate)

    if isinstance(gate, cirq.MatrixGate):
        return np.allclose(abs(cirq.unitary(gate).diagonal()), 1.0, rtol=rtol, atol=atol)

    return False


def is_clifford(circuit: cirq.Circuit) -> bool:
    """Checks if circuit is Clifford, or can be simulated with a stabilizer simulator.

    Args:
        circuit: Circuit to be checked.

    Returns:
        A boolean value representing whether `circuit` contains only Clifford gates.
    """
    for op in circuit.all_operations():
        op = op.without_classical_controls()
        if (
            not cirq.has_stabilizer_effect(op)
            and not isinstance(
                op.gate,
                (
                    cirq.DepolarizingChannel,
                    cirq.AsymmetricDepolarizingChannel,
                    cirq.BitFlipChannel,
                    cirq.PhaseFlipChannel,
                    cirq.PhaseDampingChannel,
                    cirq.ResetChannel,
                    cirq.MeasurementGate,
                    cirq.PauliMeasurementGate,
                ),
            )
            and not (
                isinstance(op.gate, cirq.RandomGateChannel)
                and cirq.has_stabilizer_effect(op.gate.sub_gate)
            )
        ):
            return False

    return True


def is_blocking(op: cirq.Operation) -> bool:
    """Checks if gate is a barrier or has no unitary representation.

    Args:
        op: Gate to be checked.

    Returns:
        A boolean value representing whether the gate is a blocking gate.
    """
    return isinstance(op.gate, css.Barrier) or not cirq.has_unitary(op)


def expressible_with_single_qubit_gates(op: cirq.Operation) -> bool:
    """Checks if an operation can be decomposed into operations on at most one qubit.

    Args:
        op: The operation to check.

    Returns:
        Whether or not `op` is a single-qubit gate, or can be deconstructed into single-qubit gates.
    """
    if isinstance(op.gate, (cirq.IdentityGate, cirq.ParallelGate, cirq.MeasurementGate)):
        return True
    if isinstance(op.gate, css.ParallelGates):
        return all(expressible_with_single_qubit_gates(o) for o in cirq.decompose_once(op))
    return cirq.num_qubits(op) <= 1


def interaction_graph(circuit: cirq.AbstractCircuit) -> nx.Graph:
    """Constructs a graph representing the qubits and interactions in `circuit`.

    Only two-qubit interactions are considered; gates on three or more qubits are ignored.

    Args:
        circuit: The circuit for which to construct the interaction graph.

    Returns:
        A graph containing all qubits in `circuit`, with edges between each pair which interact
        with a (non-separable) two-qubit gate.
    """
    graph = nx.Graph()
    for op in circuit.all_operations():
        graph.add_nodes_from(op.qubits)
        if cirq.num_qubits(op) == 2 and not expressible_with_single_qubit_gates(op):
            graph.add_edge(*op.qubits)

    return graph


def operations_to_unitary(
    ops: Iterable[cirq.Operation], qubits: Sequence[cirq.Qid]
) -> npt.NDArray[np.complex128]:
    """Compute the unitary effect of a series of operations.

    This function is equivalent to `cirq.Circuit(ops).unitary(qubits)`, but significantly faster
    because it avoids instantiating the circuit.

    Args:
        ops: The sequence of operations from which to compute a unitary.
        qubits: The way qubits should be ordered in the unitary matrix. Must contain all qubits
            present in `ops`. If larger, the resulting unitary will be expanded as if an identity
            was applied to the extra qubits.

    Returns:
        The unitary matrix computed from the product of the unitary effects of each operation.

    Raises:
        ValueError: If the resulting matrix could not be computed (likely due to a nonunitary
            operation being passed as part of `ops`).
    """
    dimensions = [q.dimension for q in qubits]
    size = int(np.prod(dimensions))
    state = np.eye(size, dtype=complex).reshape(*dimensions, *dimensions)
    args = cirq.ApplyUnitaryArgs(state, state.copy(), range(len(qubits)))
    mat = cirq.apply_unitaries(ops, qubits, args, default=None)

    if mat is None:
        raise ValueError("Operations have a nonunitary effect")

    return mat.reshape(size, size)
