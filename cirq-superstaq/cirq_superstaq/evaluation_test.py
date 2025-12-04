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

import math

import cirq
import general_superstaq as gss
import numpy as np
import pytest
import sympy

import cirq_superstaq as css


def test_paramval_to_float() -> None:
    expr = sympy.sqrt(8)
    assert css.evaluation.paramval_to_float(expr) == math.sqrt(8)

    expr = sympy.var("x")
    with pytest.raises(gss.SuperstaqException, match=r"We don't support parametrized circuits."):
        css.evaluation.paramval_to_float(expr)


def test_is_known_diagonal() -> None:
    assert css.evaluation.is_known_diagonal(cirq.Z**1.1)
    assert css.evaluation.is_known_diagonal(cirq.CCZ.controlled())
    assert css.evaluation.is_known_diagonal(cirq.MatrixGate(np.diag([1, -1, 1j, -1j])))
    assert not css.evaluation.is_known_diagonal(cirq.X)


def test_is_clifford() -> None:
    qubits = cirq.LineQubit.range(2)
    non_clifford_circuit = cirq.Circuit()
    non_clifford_circuit.append(cirq.T(qubits[0]))
    assert not css.evaluation.is_clifford(non_clifford_circuit)

    clifford_circuit = cirq.Circuit()
    clifford_circuit.append(cirq.X(qubits[0]))
    assert css.evaluation.is_clifford(clifford_circuit)

    clifford_noise_channel = cirq.DepolarizingChannel(p=0.01, n_qubits=1)
    assert css.evaluation.is_clifford(clifford_circuit.with_noise(clifford_noise_channel))

    non_clifford_channel = cirq.amplitude_damp(0.1)
    assert not css.evaluation.is_clifford(clifford_circuit.with_noise(non_clifford_channel))

    random_clifford = cirq.S.with_probability(0.1)
    assert css.evaluation.is_clifford(clifford_circuit.with_noise(random_clifford))

    random_nonclifford = cirq.T.with_probability(0.1)
    assert not css.evaluation.is_clifford(clifford_circuit.with_noise(random_nonclifford))


def test_is_blocking() -> None:
    q0, q1 = cirq.LineQubit.range(2)

    assert not css.evaluation.is_blocking(cirq.I(q0))
    assert not css.evaluation.is_blocking(cirq.X(q1))
    assert not css.evaluation.is_blocking(cirq.CZ(q1, q0))

    assert css.evaluation.is_blocking(css.Barrier(1).on(q0))
    assert css.evaluation.is_blocking(css.Barrier(2).on(q0, q1))
    assert css.evaluation.is_blocking(cirq.measure(q0))
    assert css.evaluation.is_blocking(cirq.measure(q0, q1))
    assert css.evaluation.is_blocking(cirq.ResetChannel(2).on(q0))


@pytest.mark.parametrize(
    "gate",
    [
        cirq.I,
        cirq.X,
        cirq.ry(1.2),
        cirq.GlobalPhaseGate(-1j),
        cirq.GlobalPhaseGate(-1j).controlled(1),
        cirq.ParallelGate(cirq.Y, 2),
        cirq.MatrixGate(cirq.testing.random_unitary(2)),
        cirq.depolarize(0.1),
        cirq.IdentityGate(3),
        cirq.IdentityGate(qid_shape=(3, 7, 2)),
        cirq.MatrixGate(cirq.testing.random_unitary(3), qid_shape=(3,)),
        css.Barrier(3),
        css.Barrier(qid_shape=(3, 7, 2)),
        css.ParallelRGate(1.1, 2.2, 3),
        css.ParallelGates(cirq.X, cirq.rz(1.2)),
        css.ParallelGates(cirq.XPowGate(dimension=3), cirq.ZPowGate(dimension=3)),
        cirq.MeasurementGate(2),
        cirq.MeasurementGate(qid_shape=(3, 7, 2)),
    ],
)
def test_is_separable(gate: cirq.Gate) -> None:
    qubits = cirq.LineQid.for_gate(gate)
    assert css.evaluation.expressible_with_single_qubit_gates(gate(*qubits))


@pytest.mark.parametrize(
    "gate",
    [
        cirq.CX,
        cirq.CCZ,
        cirq.ry(1.2).controlled(1),
        cirq.GlobalPhaseGate(-1j).controlled(2),
        cirq.MatrixGate(cirq.testing.random_unitary(4)),
        cirq.MatrixGate(cirq.testing.random_unitary(9), qid_shape=(3, 3)),
        css.ParallelGates(cirq.CX, cirq.rz(1.2)),
    ],
)
def test_expressible_with_single_qubit_gates(gate: cirq.Gate) -> None:
    qubits = cirq.LineQid.for_gate(gate)
    assert not css.evaluation.expressible_with_single_qubit_gates(gate(*qubits))


def test_interaction_graph() -> None:
    q0, q1, q2, q3, q4 = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        cirq.CX(q0, q1),
        css.barrier(q0, q2),
        cirq.X.on_each(q0, q3),
        cirq.parallel_gate_op(cirq.Y, q1, q2),
        css.ParallelGates(cirq.X, cirq.Y).on(q1, q3),
        cirq.ry(1.2).on(q2).controlled_by(q3),
        cirq.ry(1.2).on(q0).controlled_by(q1, q4),
        cirq.CCX(q1, q2, q4),
        css.ParallelRGate(1.1, 2.2, 3).on(q0, q2, q4),
    )
    graph = css.evaluation.interaction_graph(circuit)
    assert set(graph.nodes) == {q0, q1, q2, q3, q4}
    assert len(graph.edges) == 2
    assert graph.has_edge(q0, q1)
    assert graph.has_edge(q2, q3)


def test_operations_to_unitary() -> None:
    q2 = cirq.LineQubit(2)
    q4 = cirq.LineQubit(4)
    q5 = cirq.LineQubit(5)

    circuit = cirq.testing.random_circuit([q2, q4, q5], 10, 1)

    mat = css.evaluation.operations_to_unitary(circuit.all_operations(), [q2, q4, q5])
    assert np.allclose(mat, circuit.unitary([q2, q4, q5]))

    mat = css.evaluation.operations_to_unitary(circuit.all_operations(), [q4, q5, q2])
    assert np.allclose(mat, circuit.unitary([q4, q5, q2]))

    with pytest.raises(ValueError, match=r"nonunitary"):
        _ = css.evaluation.operations_to_unitary([cirq.measure(q2)], [q2])


def test_operations_to_unitary_on_qids() -> None:
    q2 = cirq.LineQid(2, 3)
    q4 = cirq.LineQubit(4)
    q5 = cirq.LineQid(5, 4)

    circuit = cirq.Circuit(
        cirq.MatrixGate(cirq.testing.random_unitary(3), qid_shape=(3,)).on(q2),
        cirq.MatrixGate(cirq.testing.random_unitary(2), qid_shape=(2,)).on(q4),
        cirq.MatrixGate(cirq.testing.random_unitary(4), qid_shape=(4,)).on(q5),
        cirq.MatrixGate(cirq.testing.random_unitary(6), qid_shape=(3, 2)).on(q2, q4),
        cirq.MatrixGate(cirq.testing.random_unitary(8), qid_shape=(4, 2)).on(q5, q4),
        cirq.MatrixGate(cirq.testing.random_unitary(12), qid_shape=(4, 3)).on(q5, q2),
        cirq.MatrixGate(cirq.testing.random_unitary(24), qid_shape=(2, 4, 3)).on(q4, q5, q2),
    )

    mat = css.evaluation.operations_to_unitary(circuit.all_operations(), [q2, q4, q5])
    assert np.allclose(mat, circuit.unitary([q2, q4, q5]))

    mat = css.evaluation.operations_to_unitary(circuit.all_operations(), [q4, q5, q2])
    assert np.allclose(mat, circuit.unitary([q4, q5, q2]))
