import abc
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Tuple, Type

import cirq
import numpy as np
import numpy.typing as npt
from cirq.ops.common_gates import proper_repr

import cirq_superstaq as css


class BSwapPowGate(cirq.EigenGate):
    """iSWAP-like qutrit entangling gate swapping the "02" and "11" states of two qutrits."""

    @property
    def dimension(self) -> int:
        return 3

    @property
    def _swapped_states(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (0, 2), (1, 1)

    def _qid_shape_(self) -> Tuple[int, int]:
        return self.dimension, self.dimension

    def _swapped_state_indices(self) -> Tuple[int, int]:
        return (
            cirq.big_endian_digits_to_int(self._swapped_states[0], base=self._qid_shape_()),
            cirq.big_endian_digits_to_int(self._swapped_states[1], base=self._qid_shape_()),
        )

    def _eigen_shifts(self) -> List[float]:
        return [0.0, 0.5, -0.5]

    def _eigen_components(self) -> List[Tuple[float, npt.NDArray[np.float_]]]:
        idx0, idx1 = self._swapped_state_indices()

        d = self.dimension**2

        projector_p = np.zeros((d, d))
        projector_p[idx0, idx0] = projector_p[idx1, idx1] = 0.5
        projector_p[idx0, idx1] = projector_p[idx1, idx0] = 0.5

        projector_n = np.zeros((d, d))
        projector_n[idx0, idx0] = projector_n[idx1, idx1] = 0.5
        projector_n[idx0, idx1] = projector_n[idx1, idx0] = -0.5

        projector_rest = np.eye(d) - projector_p - projector_n

        return [(0.0, projector_rest), (0.5, projector_p), (-0.5, projector_n)]

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        """Workaround for https://github.com/quantumlib/Cirq/issues/5980."""

        if not isinstance(other, BSwapPowGate):
            return NotImplemented

        return css.approx_eq_mod(self.exponent, other.exponent, 4.0, atol=atol)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=("BSwap", "BSwap"), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self.exponent == 1:
            return "BSWAP"
        if self.exponent == -1:
            return "BSWAP_INV"
        return f"BSWAP**{self.exponent}"

    def __repr__(self) -> str:
        if not self.global_shift:
            if self.exponent == 1:
                return "css.BSWAP"
            if self.exponent == -1:
                return "css.BSWAP_INV"
            return f"(css.BSWAP**{proper_repr(self.exponent)})"

        return (
            f"css.BSwapPowGate(exponent={proper_repr(self.exponent)}, "
            f"global_shift={proper_repr(self.global_shift)})"
        )


class QutritCZPowGate(cirq.EigenGate, cirq.InterchangeableQubitsGate):
    """For pairs of equal-dimension qudits, the generalized CZ gate is defined by the unitary:

        U = Σ_(i<d,j<d) ω**ij.|i⟩⟨i|.|j⟩⟨j|,

    where d is the dimension of the qudits and ω = exp(2πi/d).

    Currently written for qutrits (d = 3), but its implementation should work for any dimension.
    """

    @property
    def dimension(self) -> int:
        return 3

    def _qid_shape_(self) -> Tuple[int, int]:
        return self.dimension, self.dimension

    def _eigen_components(self) -> List[Tuple[float, npt.NDArray[np.float_]]]:
        eigen_components = []
        exponents = np.kron(range(self.dimension), range(self.dimension)) % self.dimension
        for x in sorted(set(exponents)):
            value = cirq.canonicalize_half_turns(2 * x / self.dimension)
            matrix = np.diag(np.asarray(exponents == x, dtype=float))
            eigen_components.append((value, matrix))

        return eigen_components

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        """Workaround for https://github.com/quantumlib/Cirq/issues/5980."""

        if isinstance(other, QutritCZPowGate):
            return css.approx_eq_mod(other.exponent, self.exponent, self.dimension)

        return NotImplemented

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.circuit_diagram_info(cirq.CZ, args)

    def __str__(self) -> str:
        if self.exponent == 1:
            return "CZ3"
        if self.exponent == -1:
            return "CZ3_INV"
        return f"CZ3**{self.exponent}"

    def __repr__(self) -> str:
        if not self.global_shift:
            if self.exponent == 1:
                return "css.CZ3"
            if self.exponent == -1:
                return "css.CZ3_INV"
            return f"(css.CZ3**{proper_repr(self.exponent)})"

        return (
            f"css.QutritCZPowGate(exponent={proper_repr(self.exponent)}, "
            f"global_shift={proper_repr(self.global_shift)})"
        )


class _QutritZPowGate(cirq.EigenGate):
    """Applies a phase rotation to a single energy level of a qutrit."""

    @property
    @abc.abstractmethod
    def _target_state(self) -> int:
        """The energy level onto which to apply a phase."""

    def _qid_shape_(self) -> Tuple[int]:
        return (3,)

    def _eigen_components(self) -> List[Tuple[float, npt.NDArray[np.float_]]]:
        d = self._qid_shape_()[0]
        projector_phase = np.zeros((d, d))
        projector_phase[self._target_state, self._target_state] = 1

        projector_rest = np.eye(d) - projector_phase
        return [(0.0, projector_rest), (1.0, projector_phase)]

    def _eigen_shifts(self) -> List[float]:
        return [0.0, 1.0]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters and self._target_state < 10:
            wire_symbol = f"Z{ord('₀') + self._target_state:c}"
        else:
            wire_symbol = f"Z[{self._target_state}]"

        return cirq.CircuitDiagramInfo(
            wire_symbols=(wire_symbol,), exponent=self._diagram_exponent(args)
        )

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        """Workaround for https://github.com/quantumlib/Cirq/issues/5980."""

        if not isinstance(other, _QutritZPowGate):
            return NotImplemented

        if other._target_state != self._target_state:
            return False

        return css.approx_eq_mod(self.exponent, other.exponent, 2.0, atol=atol)

    def __str__(self) -> str:
        if self.exponent == 1:
            return f"QutritZ{self._target_state}"
        return f"QutritZ{self._target_state}**{self.exponent}"

    def __repr__(self) -> str:
        if not self.global_shift:
            if self.exponent == 1:
                return f"css.QutritZ{self._target_state}"
            return f"(css.QutritZ{self._target_state}**{proper_repr(self.exponent)})"

        return (
            f"css.QutritZ{self._target_state}PowGate(exponent={proper_repr(self.exponent)}, "
            f"global_shift={proper_repr(self.global_shift)})"
        )


class QutritZ0PowGate(_QutritZPowGate):
    """Phase rotation on the ground state of a qutrit."""

    @property
    def _target_state(self) -> int:
        return 0


class QutritZ1PowGate(_QutritZPowGate):
    """Phase rotation on the first excited state of a qutrit."""

    @property
    def _target_state(self) -> int:
        return 1


class QutritZ2PowGate(_QutritZPowGate):
    """Phase rotation on the second excited state of a qutrit."""

    @property
    def _target_state(self) -> int:
        return 2


@cirq.value_equality(approximate=True)
class QubitSubspaceGate(cirq.Gate):
    def __init__(
        self,
        sub_gate: cirq.Gate,
        qid_shape: Sequence[int],
        subspaces: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """
        Embeds an n-qubit (i.e. SU(2^n)) gate into a given subspace of a higher-dimensional gate.

        Args:
            sub_gate: The qubit gate to promote to a higher dimension.
            qid_shape: The shape of the new gate (that is, the dimension of each Qid it acts upon).
            subspaces: If provided, the subspace (in the computational basis) of each Qid to act
                upon. By default applies to the first two levels of each Qid.

        Examples:
            QubitSubspaceGate(cirq.X, (3,)): X gate acting on levels 0 and 1 of a dimension-3 Qid.
            QubitSubspaceGate(cirq.X, (3,), [(0, 2)]): the same gate acting on levels 0 and 2.
            QubitSubspaceGate(cirq.CX, (3, 3)): CX gate on the 0-1 subspace of two dimension-3 Qids.
        """

        if subspaces is None:
            subspaces = [(0, 1)] * cirq.num_qubits(sub_gate)

        if not all(d == 2 for d in cirq.qid_shape(sub_gate)):
            raise ValueError("Only qubit gates are supported for sub_gate.")

        if not all(d >= 2 for d in qid_shape):
            raise ValueError("Invalid qid_shape (all dimensions must be at least 2).")

        if cirq.num_qubits(sub_gate) != len(qid_shape):
            raise ValueError("QubitSubspaceGate and sub_gate must have the same number of qubits.")

        if len(qid_shape) != len(subspaces) or any(len(subspace) != 2 for subspace in subspaces):
            raise ValueError("You must provide two subspace indices for every qubit.")

        self._sub_gate = sub_gate
        self._qid_shape = tuple(qid_shape)
        self._subspaces = [
            (subspace[0] % d, subspace[1] % d) for subspace, d in zip(subspaces, qid_shape)
        ]

    @property
    def sub_gate(self) -> cirq.Gate:
        return self._sub_gate

    @property
    def qid_shape(self) -> Tuple[int, ...]:
        return self._qid_shape

    @property
    def subspaces(self) -> List[Tuple[int, int]]:
        return self._subspaces

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self._sub_gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self._sub_gate)

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> "QubitSubspaceGate":
        return QubitSubspaceGate(
            cirq.resolve_parameters(self._sub_gate, resolver, recursive),
            self._qid_shape,
            self._subspaces,
        )

    def _has_unitary_(self) -> bool:
        return cirq.has_unitary(self._sub_gate)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Optional[npt.NDArray[np.complex_]]:
        if not cirq.has_unitary(self._sub_gate):
            return NotImplemented

        subspace_args = cirq.ApplyUnitaryArgs(
            target_tensor=args.target_tensor,
            available_buffer=args.available_buffer,
            axes=args.axes,
            subspaces=self._subspaces,
        )
        return cirq.apply_unitary(self._sub_gate, subspace_args)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        sub_gate_info = cirq.circuit_diagram_info(self._sub_gate, args)

        new_symbols: List[str] = []
        for symbol, subspace in zip(sub_gate_info.wire_symbols, self.subspaces):
            if args.use_unicode_characters and max(subspace) < 10:
                subspace_str = f"{ord('₀') + subspace[0]:c}{ord('₀') + subspace[1]:c}"
            else:
                subspace_str = f"[{subspace[0]},{subspace[1]}]"

            new_symbols.append(f"{symbol}{subspace_str}")

        return sub_gate_info.with_wire_symbols(new_symbols)

    def _value_equality_values_(
        self,
    ) -> Tuple[cirq.Gate, Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
        return self.sub_gate, self.qid_shape, tuple(self.subspaces)

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        if not isinstance(other, QubitSubspaceGate):
            return NotImplemented

        if cirq.qid_shape(other) != self._qid_shape:
            return False

        if other.subspaces != self.subspaces:
            return False

        return cirq.equal_up_to_global_phase(
            self.sub_gate, other.sub_gate, atol=atol
        ) or cirq.equal_up_to_global_phase(
            other.sub_gate, self.sub_gate, atol=atol
        )  # Test both orders as a workaround for https://github.com/quantumlib/Cirq/issues/5980

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["sub_gate", "qid_shape", "subspaces"])

    def __pow__(self, exponent: cirq.TParamVal) -> Optional["QubitSubspaceGate"]:
        exp_gate = cirq.pow(self._sub_gate, exponent, None)
        if exp_gate is None:
            return NotImplemented

        return QubitSubspaceGate(
            sub_gate=exp_gate, qid_shape=self._qid_shape, subspaces=self._subspaces
        )

    def __str__(self) -> str:
        if self._subspaces == [(0, 1)] * cirq.num_qubits(self):
            return f"QubitSubspaceGate({self._sub_gate}, {self._qid_shape})"

        return f"QubitSubspaceGate({self._sub_gate}, {self._qid_shape}, {self._subspaces})"

    def __repr__(self) -> str:
        if self._subspaces == [(0, 1)] * cirq.num_qubits(self):
            return f"css.QubitSubspaceGate({self._sub_gate!r}, {self._qid_shape})"

        return f"css.QubitSubspaceGate({self._sub_gate!r}, {self._qid_shape}, {self._subspaces})"


BSWAP = BSwapPowGate()
BSWAP_INV = BSwapPowGate(exponent=-1)

CZ3 = QutritCZPowGate()
CZ3_INV = QutritCZPowGate(exponent=-1)

QutritZ0 = QutritZ0PowGate()
QutritZ1 = QutritZ1PowGate()
QutritZ2 = QutritZ2PowGate()


def custom_resolver(cirq_type: str) -> Optional[Type[cirq.Gate]]:
    if cirq_type == "BSwapPowGate":
        return BSwapPowGate
    if cirq_type == "QutritCZPowGate":
        return QutritCZPowGate
    if cirq_type == "QutritZ0PowGate":
        return QutritZ0PowGate
    if cirq_type == "QutritZ1PowGate":
        return QutritZ1PowGate
    if cirq_type == "QutritZ2PowGate":
        return QutritZ2PowGate
    if cirq_type == "QubitSubspaceGate":
        return QubitSubspaceGate

    return None
