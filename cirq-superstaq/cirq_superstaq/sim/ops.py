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

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Self

import cirq
import cirq_superstaq as css
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class KrausChannel(cirq.KrausChannel):
    """Simple extension of `cirq.KrausChannel` to support qudits.

    Args:
        kraus_ops: A sequence of Kraus operators.
        qid_shape: The shape of the qudits. If None, it is inferred from the Kraus operators,
            assuming they act on qubits.
        key: The measurement key for the channel.
    """

    def __init__(
        self,
        kraus_ops: Iterable[npt.NDArray[np.complex128 | np.float64]],
        qid_shape: Sequence[int] | None = None,
        key: cirq.MeasurementKey | str | None = None,
    ) -> None:
        kraus_ops = list(kraus_ops)

        if not kraus_ops or qid_shape is None or set(qid_shape) == {2}:
            # The cirq type handles these cases already
            super().__init__(kraus_ops, key=key)
            self._qid_shape = (2,) * self._num_qubits

        else:
            dimension = np.prod(qid_shape)
            assert all(len(op) == dimension for op in kraus_ops)

            if not isinstance(key, cirq.MeasurementKey) and key is not None:
                key = cirq.MeasurementKey(key)

            self._kraus_ops = kraus_ops
            self._qid_shape = tuple(qid_shape)
            self._num_qubits = len(qid_shape)

            self._key = key

    @classmethod
    def from_channel(
        cls,
        channel: cirq.Gate,
        key: cirq.MeasurementKey | str | None = None,
        qid_shape: Sequence[int] | None = None,
    ) -> Self:
        """Creates a `KrausChannel` from a `cirq.Gate`.

        Args:
            channel: The `cirq.Gate` to convert to a `KrausChannel`.
            key: The measurement key for the new channel.
            qid_shape: The shape of the qudits for the new channel. If None, it is
                inferred from the channel.

        Returns:
            A `KrausChannel` representing the input channel.
        """
        if qid_shape is None:
            qid_shape = cirq.qid_shape(channel)

        kraus_ops: list[npt.NDArray[np.complex128]] = []
        dimension = np.prod(qid_shape)

        for kraus_op in cirq.kraus(channel):
            matrix = np.zeros((dimension, dimension), dtype=complex)
            matrix[:2, :2] = kraus_op
            kraus_ops.append(matrix)

        tot = sum(k.T.conj() @ k for k in kraus_ops)
        k0 = np.sqrt(np.eye(np.shape(tot)[0]) - tot)
        if not np.allclose(k0, 0.0, atol=2e-8):
            kraus_ops.append(k0)
        return cls(kraus_ops, qid_shape=qid_shape, key=key)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbol = f"Kraus({'x'.join(str(d) for d in self._qid_shape)})"
        return cirq.CircuitDiagramInfo(wire_symbols=(wire_symbol,))

    def _qid_shape_(self) -> tuple[int, ...]:
        return self._qid_shape


@cirq.value_equality
class JumpChannel(cirq.Gate):
    """A channel that represents jumps between energy levels.

    Args:
        transition_matrix: A matrix of probabilities, where `transition_matrix[i, j]` is the
            probability of jumping from level `j` to level `i`.
    """

    def __init__(self, transition_matrix: npt.ArrayLike) -> None:
        transition_matrix = np.asarray(transition_matrix)

        # Make square
        max_level = max(transition_matrix.shape)
        self._transition_matrix = np.zeros((max_level, max_level))
        self._transition_matrix[: transition_matrix.shape[0], : transition_matrix.shape[1]] = (
            transition_matrix
        )

    def _transition_matrix_(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self._transition_matrix

    @property
    def max_level(self) -> int:
        """The maximum energy level."""
        return max(self._transition_matrix.shape)

    def _qid_shape_(self) -> tuple[int]:
        return (2,)

    def _value_equality_values_(self) -> tuple[tuple[float, ...], ...]:
        return tuple(map(tuple, self._transition_matrix))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbol = f"Jump({self.max_level})"
        return cirq.CircuitDiagramInfo(wire_symbols=[wire_symbol])

    def then(self, other: JumpChannel) -> JumpChannel:
        """Constructs a new `JumpChannel`, equivalent to this channel followed by another.

        Args:
            other: A `JumpChannel` which occurs immediately after this.

        Returns:
            A single `JumpChannel` equivalent to the combined channels.
        """
        shape = np.maximum(
            self._transition_matrix.shape,
            other._transition_matrix.shape,
        )
        tmat_self = np.pad(
            self._transition_matrix,
            (
                (0, shape[0] - self._transition_matrix.shape[0]),
                (0, shape[1] - self._transition_matrix.shape[1]),
            ),
        )
        tmat_other = np.pad(
            other._transition_matrix,
            (
                (0, shape[0] - other._transition_matrix.shape[0]),
                (0, shape[1] - other._transition_matrix.shape[1]),
            ),
        )

        m0 = tmat_other @ tmat_self
        m1 = tmat_other * (1 - tmat_self.sum(0, keepdims=True))
        m2 = tmat_self * (1 - tmat_other.sum(0, keepdims=True).T)
        return JumpChannel(m0 + m1 + m2)

    def to_kraus_channel(
        self,
        dimension: int | None = None,
        key: cirq.MeasurementKey | str | None = None,
    ) -> KrausChannel:
        """Converts this `JumpChannel` to a `KrausChannel`.

        Args:
            dimension: The dimension of the qudit. If None, it is inferred from
                the jump probabilities.
            key: The measurement key for the new channel.

        Returns:
            The equivalent `KrausChannel`.
        """
        dimension = dimension or self.max_level
        transition_matrix = np.asarray(self._transition_matrix)
        jump_ops = []
        for idxs, p in np.ndenumerate(transition_matrix):
            if p:
                k = np.zeros((dimension, dimension))
                k[idxs] = np.sqrt(p)
                jump_ops.append(k)
        tot = sum(k.T.conj() @ k for k in jump_ops)
        kraus_ops = [k[:dimension] for k in jump_ops if np.any(k)]
        k0 = np.sqrt((np.eye(np.shape(tot)[0]) - tot).clip(0, 1))
        if np.any(k0):
            kraus_ops.append(k0)
        return KrausChannel(kraus_ops, qid_shape=(dimension,), key=key)


class QuditPermutationGate(cirq.QubitPermutationGate):
    """A gate that permutes qudits.

    Args:
        permutation: A list of integers representing the permutation of qudits.
        dimension: The dimension of the qudits.
    """

    def __init__(self, permutation: Sequence[int], dimension: int = 2) -> None:
        super().__init__(permutation)
        self._dimension = dimension

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._dimension,) * len(self._permutation)

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> Iterator[cirq.Operation]:
        base_qubits = [q.with_dimension(2) for q in qubits]
        qubit_map = {bq: q for bq, q in zip(base_qubits, qubits)}
        for op in super()._decompose_(base_qubits):
            assert isinstance(op, cirq.Operation)
            qs = [qubit_map[q] for q in op.qubits]
            yield css.qudit_swap_op(*qs)


def qudit_permutation_op(permutation: Sequence[int], *qubits: cirq.Qid) -> cirq.Operation:
    """Creates a permutation operation on the given qudits.

    Args:
        permutation: The permutation to apply to the qudits.
        *qubits: The qudits to permute.

    Returns:
        A `cirq.Operation` that permutes the qudits.
    """
    dimension = max(q.dimension for q in qubits)
    if dimension == 2:
        return cirq.QubitPermutationGate(permutation).on(*qubits)

    return QuditPermutationGate(permutation, dimension=dimension).on(*qubits)


def with_dimension(circuit: cirq.Circuit, dimension: int) -> cirq.Circuit:
    """Expands all operations in `circuit` to have the given `dimension`.

    Args:
        circuit: The circuit to expand.
        dimension: The target dimension for all qudits in the circuit.

    Returns:
        A new `cirq.Circuit` with all operations expanded to the given dimension.
    """

    def _map_fn(op: cirq.Operation) -> cirq.Operation:
        if cirq.qid_shape(op) == (dimension,) * cirq.num_qubits(op):
            return op

        qudits = [q.with_dimension(dimension) for q in op.qubits]

        if isinstance(op.gate, cirq.MeasurementGate):
            gate = cirq.MeasurementGate(
                qid_shape=(dimension,) * cirq.num_qubits(op),
                key=op.gate.mkey,
                invert_mask=op.gate.invert_mask,
                confusion_map=op.gate.confusion_map,
            )
            return gate.on(*qudits)

        if isinstance(op.gate, cirq.QubitPermutationGate):
            return qudit_permutation_op(op.gate.permutation, *qudits)

        if isinstance(op.gate, css.Barrier):
            return css.barrier(*qudits)

        if isinstance(op.gate, JumpChannel):
            return op.gate.to_kraus_channel(dimension=dimension).on(*qudits)

        if len(qudits) < 8 and cirq.has_kraus(op) and not cirq.has_unitary(op):
            assert op.gate is not None
            qid_shape = (dimension,) * len(qudits)
            return KrausChannel.from_channel(op.gate, qid_shape=qid_shape).on(*qudits)

        return css.qubit_subspace_op(op, [dimension] * len(qudits))

    return circuit.map_operations(_map_fn)
