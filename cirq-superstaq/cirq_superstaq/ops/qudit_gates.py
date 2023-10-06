from __future__ import annotations

import abc
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import cirq
import numpy as np
import numpy.typing as npt
from cirq.ops.common_gates import proper_repr

import cirq_superstaq as css


@cirq.value_equality
class QuditSwapGate(cirq.Gate, cirq.InterchangeableQubitsGate):
    """A (non-parametrized) SWAP gate on two qudits of arbitrary dimension."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The qudit dimension on which this SWAP gate will act.

        Returns:
            The qudit dimension.
        """
        return self._dimension

    def _qid_shape_(self) -> Tuple[int, int]:
        return self.dimension, self.dimension

    def _value_equality_values_(self) -> int:
        return self.dimension

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        if isinstance(other, QuditSwapGate):
            return other.dimension == self.dimension

        elif self.dimension == 2:
            return cirq.equal_up_to_global_phase(other, cirq.SWAP) or cirq.equal_up_to_global_phase(
                cirq.SWAP, other
            )

        return NotImplemented

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Optional[npt.NDArray[np.complex_]]:
        for i in range(self._dimension):
            for j in range(i):
                idx0 = args.subspace_index(i * self._dimension + j)
                idx1 = args.subspace_index(j * self._dimension + i)
                args.available_buffer[idx0] = args.target_tensor[idx0]
                args.target_tensor[idx0] = args.target_tensor[idx1]
                args.target_tensor[idx1] = args.available_buffer[idx0]

        return args.target_tensor

    def _trace_distance_bound_(self) -> float:
        return 1.0

    def _is_parameterized_(self) -> bool:
        return False

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.circuit_diagram_info(cirq.SWAP, args)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["dimension"])

    def __pow__(
        self, exponent: cirq.TParamVal
    ) -> Optional[Union[QuditSwapGate, cirq.IdentityGate]]:
        if not cirq.is_parameterized(exponent):
            if exponent % 2 == 1:
                return self
            if exponent % 2 == 0:
                return cirq.IdentityGate(qid_shape=self._qid_shape_())

        return NotImplemented

    def __str__(self) -> str:
        return f"SWAP{self._dimension}"

    def __repr__(self) -> str:
        if self._dimension == 3:
            return f"css.SWAP{self._dimension}"

        return f"css.QuditSwapGate(dimension={self._dimension!r})"


class BSwapPowGate(cirq.EigenGate, cirq.InterchangeableQubitsGate):
    """iSWAP-like qutrit entangling gate swapping the "11" and "22" states of two qutrits."""

    @property
    def dimension(self) -> int:
        """Indicates that this gate acts on qutrits.

        Returns:
            The integer `3`, representing the qudit dimension for qutrits.
        """
        return 3

    @property
    def _swapped_states(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (1, 1), (2, 2)

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
        """Indicates that this gate acts on qutrits.

        Returns:
            The integer `3`, representing the qudit dimension for qutrits.
        """
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
        return cirq.CircuitDiagramInfo(
            wire_symbols=("@", "@"), exponent=self._diagram_exponent(args)
        )

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


class VirtualZPowGate(cirq.EigenGate):
    """Applies a phase rotation between two successive energy levels of a qudit."""

    def __init__(
        self,
        dimension: int = 2,
        level: int = 1,
        exponent: cirq.TParamVal = 1.0,
        global_shift: float = 0.0,
    ) -> None:
        """Initializes as `VirtualZPowGate`.

        Args:
            dimension: The qudit dimension on which this gate will act.
            level: The lowest energy level onto which this gate applies a phase. For example,
                passing `level=2` a phase of `(-1)**exponent` will be applied to energy levels
                `[2, ..., dimension - 1]`. This is equivalent to phase shifting all subsequent
                single-qudit gates acting in the `(1, 2)` subspace (assuming all other gates commute
                with this one).
            exponent: This gate's exponent (see `cirq.EigenGate` documentation for details).
            global_shift: This gate's global phase (see `cirq.EigenGate` documentation for details).

        Raises:
            ValueError: If `dimension` is less than two.
            ValueError: If `level` is invalid for the given dimension.
        """
        if dimension < 2:
            raise ValueError("Invalid dimension (must be at least 2).")

        # Allow e.g. level=-1 to specify the highest energy level
        if not 0 < abs(level) < dimension:
            raise ValueError(f"Invalid energy level for a dimension-{dimension} gate.")

        self._dimension = dimension
        self._level = level % dimension
        super().__init__(exponent=exponent, global_shift=global_shift)

    @property
    def dimension(self) -> int:
        """The qudit dimension on which this gate acts."""
        return self._dimension

    @property
    def level(self) -> int:
        """The lowest energy level onto which this gate applies a phase; for example if `level=2`
        a phase of `(-1)**exponent` will be applied to energy levels `[2, ..., dimension - 1]`. This
        is equivalent to phase shifting all subsequent single-qudit gates acting in the `(1, 2)`
        subspace (assuming all other gates commute with this one).
        """
        return self._level

    def _qid_shape_(self) -> Tuple[int]:
        return (self._dimension,)

    def _with_exponent(self, exponent: cirq.TParamVal) -> VirtualZPowGate:
        return VirtualZPowGate(
            dimension=self._dimension,
            level=self._level,
            exponent=exponent,
            global_shift=self._global_shift,
        )

    def _eigen_components(self) -> List[Tuple[float, npt.NDArray[np.float_]]]:
        projector_phase = np.zeros(self._dimension)
        projector_phase[self._level :] = 1
        projector_rest = 1 - projector_phase
        return [(0.0, np.diag(projector_rest)), (1.0, np.diag(projector_phase))]

    def _eigen_shifts(self) -> List[float]:
        return [0.0, 1.0]

    def _value_equality_values_(self) -> tuple[object, ...]:
        return (*super()._value_equality_values_(), self._dimension, self._level)

    def _value_equality_approximate_values_(self) -> tuple[object, ...]:
        return (*super()._value_equality_approximate_values_(), self._dimension, self._level)

    def _equal_up_to_global_phase_(self, other: object, atol: float) -> Optional[bool]:
        if not isinstance(other, VirtualZPowGate):
            return NotImplemented

        if self._dimension != other.dimension or self._level != other.level:
            return False

        return css.approx_eq_mod(self._exponent, other.exponent, 2.0, atol=atol)

    def _json_dict_(self) -> dict[str, object]:
        return cirq.obj_to_dict_helper(self, ["exponent", "global_shift", "dimension", "level"])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters and self._level < 10:
            wire_symbol = f"VZ{ord('₀') + self._level:c}₊"
        else:
            wire_symbol = f"VZ({self._level}+)"

        return cirq.CircuitDiagramInfo(
            wire_symbols=(wire_symbol,), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        base_str = f"VZ({self._level}+)"
        if self._exponent == 1:
            return base_str
        return f"{base_str}**{self._exponent}"

    def __repr__(self) -> str:
        args = [f"dimension={self._dimension}"]
        if self._level != 1:
            args.append(f"level={self._level}")
        if self._exponent != 1:
            args.append(f"exponent={proper_repr(self._exponent)}")
        if self._global_shift:
            args.append(f"global_shift={self._global_shift!r}")

        return "css.VirtualZPowGate(" + ", ".join(args) + ")"


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
    """Embeds an n-qubit (i.e. SU(2^n)) gate into a given subspace of a higher-dimensional gate."""

    def __init__(
        self,
        sub_gate: cirq.Gate,
        qid_shape: Sequence[int],
        subspaces: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """Initializes a `QubitSubspaceGate`.

        Args:
            sub_gate: The qubit gate to promote to a higher dimension.
            qid_shape: The shape of the new gate (that is, the dimension of each Qid it acts upon).
            subspaces: If provided, the subspace (in the computational basis) of each Qid to act
                upon. By default applies to the first two levels of each Qid.

        Examples:
            `QubitSubspaceGate(cirq.X, (3,))`: An X gate acting on levels 0 and 1 of a dimension-3
                Qid.
            `QubitSubspaceGate(cirq.X, (3,), [(0, 2)])`: The same gate acting on levels 0 and 2.
            `QubitSubspaceGate(cirq.CX, (3, 3))`: A CX gate on the 0-1 subspace of two dimension-3
                Qids.
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
        """The gate that is applied to the specified subspace.

        Returns:
            The underlying gate used.
        """
        return self._sub_gate

    @property
    def qid_shape(self) -> Tuple[int, ...]:
        """Specifies the qudit dimension for each of the inputs.

        Returns:
            The dimensions for the input qudits.
        """
        return self._qid_shape

    @property
    def subspaces(self) -> List[Tuple[int, int]]:
        """A list of subspace indices acted upon.

        For instance, a CX on the 0-1 qubit subspace of two qudits would have subspaces of
        [(0, 1), (0, 1)]. The same gate acting on the 1-2 subspaces of both qudits would correspond
        to [(1, 2), (1, 2)].

        Returns:
            A list of dimensions tuples, specified for each subspace.
        """
        return self._subspaces

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self._sub_gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self._sub_gate)

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> QubitSubspaceGate:
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

        # Do not ignore global phase when comparing sub gates, as it becomes physical when the gate
        # is expanded to higher dimensions
        return cirq.approx_eq(self.sub_gate, other.sub_gate, atol=atol)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["sub_gate", "qid_shape", "subspaces"])

    def __pow__(self, exponent: cirq.TParamVal) -> Optional[QubitSubspaceGate]:
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


def qudit_swap_op(qudit0: cirq.Qid, qudit1: cirq.Qid) -> cirq.Operation:
    """Construct a SWAP gate and apply it to the provided qudits.

    If both qudits have dimension 2, uses `cirq.SWAP`; otherwise uses `QuditSwapGate`.

    Args:
        qudit0: The first qudit to swap.
        qudit1: The second qudit to swap.

    Returns:
        A SWAP gate acting on the provided qudits.

    Raises:
        ValueError: If the input qudits don't have the same dimension.
    """

    if qudit0.dimension != qudit1.dimension:
        raise ValueError(f"{qudit0} and {qudit1} do not have the same dimension.")

    if qudit0.dimension == 2:
        return cirq.SWAP(qudit0, qudit1)

    return QuditSwapGate(dimension=qudit0.dimension).on(qudit0, qudit1)


def qubit_subspace_op(
    sub_op: cirq.Operation,
    qid_shape: Sequence[int],
    subspaces: Optional[Sequence[Tuple[int, int]]] = None,
) -> cirq.Operation:
    """Embeds a qubit Operation into a given subspace of a higher-dimensional Operation.

    Uses `QubitSubspaceGate`.

    Args:
        sub_op: The `cirq.Operation` to embed.
        qid_shape: The dimensions of the subspace.
        subspaces: The list of all subspaces.

    Returns:
        A `cirq.Operation` embedding a low-dimensional operation.

    Raises:
        ValueError: If there is no gate specified for the subspace operation.
    """
    if not sub_op.gate:
        raise ValueError(f"{sub_op} has no gate.")

    qudits = [qubit.with_dimension(d) for qubit, d in zip(sub_op.qubits, qid_shape)]
    return QubitSubspaceGate(sub_op.gate, qid_shape, subspaces=subspaces).on(*qudits)


SWAP3 = QuditSwapGate(dimension=3)

BSWAP = BSwapPowGate()
BSWAP_INV = BSwapPowGate(exponent=-1)

CZ3 = QutritCZPowGate()
CZ3_INV = QutritCZPowGate(exponent=-1)

QutritZ0 = QutritZ0PowGate()
QutritZ1 = QutritZ1PowGate()
QutritZ2 = QutritZ2PowGate()


def custom_resolver(
    cirq_type: str,
) -> Optional[Type[cirq.Gate]]:
    """Tells `cirq.json` how to deserialize cirq_superstaq's custom gates.

    Changes to gate names in this file should be reflected in this resolver.
    See quantumai.google/cirq/dev/serialization for more information about (de)serialization.

    Args:
        cirq_type: The string of the gate type for the serializer to resolve.

    Returns:
        The resolved Cirq Gate matching the input, or None if no match.
    """
    if cirq_type == "QuditSwapGate":
        return QuditSwapGate
    if cirq_type == "BSwapPowGate":
        return BSwapPowGate
    if cirq_type == "QutritCZPowGate":
        return QutritCZPowGate
    if cirq_type == "VirtualZPowGate":
        return VirtualZPowGate
    if cirq_type == "QutritZ0PowGate":
        return QutritZ0PowGate
    if cirq_type == "QutritZ1PowGate":
        return QutritZ1PowGate
    if cirq_type == "QutritZ2PowGate":
        return QutritZ2PowGate
    if cirq_type == "QubitSubspaceGate":
        return QubitSubspaceGate

    return None
