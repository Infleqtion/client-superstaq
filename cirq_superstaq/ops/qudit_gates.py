from typing import Any, List, Optional, Tuple, Type

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


BSWAP = BSwapPowGate()
BSWAP_INV = BSwapPowGate(exponent=-1)

CZ3 = QutritCZPowGate()
CZ3_INV = QutritCZPowGate(exponent=-1)


def custom_resolver(cirq_type: str) -> Optional[Type[cirq.Gate]]:
    if cirq_type == "BSwapPowGate":
        return BSwapPowGate
    if cirq_type == "QutritCZPowGate":
        return QutritCZPowGate

    return None
