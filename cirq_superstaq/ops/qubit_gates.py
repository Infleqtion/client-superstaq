"""Miscellaneous custom gates that we encounter and want to explicitly define."""

from typing import AbstractSet, Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import cirq
import numpy as np
import numpy.typing as npt
from cirq.ops.common_gates import _pi

import cirq_superstaq as css


def approx_eq_mod(a: cirq.TParamVal, b: cirq.TParamVal, period: float, atol: float = 1e-8) -> bool:
    """Check if a ~= b (mod period). If either input is an unresolved parameter, returns a == b."""

    if cirq.is_parameterized(a) or cirq.is_parameterized(b):
        return a == b

    return cirq.all_near_zero_mod(a - b, period, atol=atol)


@cirq.value_equality(approximate=True)
class ZZSwapGate(cirq.Gate, cirq.ops.gate_features.InterchangeableQubitsGate):
    r"""The ZZ-SWAP gate, which performs the ZZ-interaction followed by a SWAP.

    ZZ-SWAPs are useful for applications like QAOA or Hamiltonian Simulation,
    particularly on linear- or low- connectivity devices. See https://arxiv.org/pdf/2004.14970.pdf
    for an application of ZZ SWAP networks.

    The unitary for a ZZ-SWAP gate parametrized by ZZ-interaction angle :math:`\theta` is:

     .. math::

        \begin{bmatrix}
        1 & . & . & . \\
        . & . & e^{i \theta} & . \\
        . & e^{i \theta} & . & . \\
        . & . & . & 1 \\
        \end{bmatrix}

    where '.' means '0'.
    For :math:`\theta = 0`, the ZZ-SWAP gate is just an ordinary SWAP.
    """

    def __init__(self, theta: cirq.TParamVal) -> None:
        """
        Args:
            theta: ZZ-interaction angle in radians
        """
        self.theta = theta

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> Optional[npt.NDArray[np.complex_]]:
        if self._is_parameterized_():
            return None
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, np.exp(1j * self.theta), 0],
                [0, np.exp(1j * self.theta), 0, 0],
                [0, 0, 0, 1],
            ]
        )

    def _value_equality_values_(self) -> cirq.TParamVal:
        if cirq.is_parameterized(self.theta):
            return self.theta

        return self.theta % (2 * np.pi)

    def _value_equality_approximate_values_(self) -> cirq.PeriodicValue:
        return cirq.PeriodicValue(self.theta, 2 * np.pi)

    def __pow__(
        self, exponent: cirq.TParamVal
    ) -> Union["ZZSwapGate", cirq.ZZPowGate, cirq.type_workarounds.NotImplementedType]:
        if exponent % 2 == 1:
            return ZZSwapGate(exponent * self.theta)
        if exponent % 2 == 0:
            return cirq.ZZPowGate(exponent=exponent * self.theta / _pi(self.theta))
        return NotImplemented

    def __str__(self) -> str:
        return f"ZZSwapGate({self.theta})"

    def __repr__(self) -> str:
        return f"css.ZZSwapGate({self.theta})"

    def _decompose_(self, qubits: Tuple[cirq.Qid, cirq.Qid]) -> Iterator[cirq.Operation]:
        yield cirq.CX(qubits[0], qubits[1])
        yield cirq.CX(qubits[1], qubits[0])
        yield cirq.Z(qubits[1]) ** (self.theta / _pi(self.theta))
        yield cirq.CX(qubits[0], qubits[1])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        t = args.format_radians(self.theta)
        return cirq.CircuitDiagramInfo(wire_symbols=(f"ZZSwap({t})", f"ZZSwap({t})"))

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.theta)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.theta)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> "ZZSwapGate":
        return ZZSwapGate(
            cirq.resolve_parameters(self.theta, resolver, recursive),
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> npt.NDArray[np.complex_]:
        zo = args.subspace_index(0b01)
        oz = args.subspace_index(0b10)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        args.target_tensor[zo] *= np.exp(1j * self.theta)
        args.target_tensor[oz] *= np.exp(1j * self.theta)
        return args.target_tensor

    def _pauli_expansion_(
        self,
    ) -> Union[cirq.value.LinearDict[str], cirq.type_workarounds.NotImplementedType]:
        if cirq.is_parameterized(self):
            return NotImplemented
        return cirq.value.LinearDict(
            {
                "II": 0.5,
                "XX": 0.5 * np.exp(1j * self.theta),
                "YY": 0.5 * np.exp(1j * self.theta),
                "ZZ": 0.5,
            }
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, cirq.Qid]) -> Optional[str]:
        if approx_eq_mod(self.theta, 0.0, 2 * np.pi):
            return cirq.SWAP._qasm_(args, qubits)

        return args.format(
            "zzswap({0:half_turns}) {1},{2};\n",
            self.theta / np.pi,
            qubits[0],
            qubits[1],
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["theta"])


class ZXPowGate(cirq.EigenGate, cirq.Gate):
    r"""The ZX-parity gate, possibly raised to a power.

    Per arxiv.org/pdf/1904.06560v3 eq. 135, the ZX**t gate implements the following unitary:

    .. math::

        e^{-\frac{i\pi}{2} t Z \otimes X} = \begin{bmatrix}
                                        c & -s & . & . \\
                                        -s & c & . & . \\
                                        . & . & c & s \\
                                        . & . & s & c \\
                                        \end{bmatrix}

    where '.' means '0' and :math:`c = \cos(\frac{\pi t}{2})`
    and :math:`s = i \sin(\frac{\pi t}{2})`.

    """

    def _eigen_components(self) -> List[Tuple[float, npt.NDArray[np.float_]]]:
        return [
            (
                0.0,
                np.array(
                    [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, -0.5], [0, 0, -0.5, 0.5]]
                ),
            ),
            (
                1.0,
                np.array(
                    [[0.5, -0.5, 0, 0], [-0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]
                ),
            ),
        ]

    def _eigen_shifts(self) -> List[float]:
        return [0, 1]

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=("Z", "X"), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        return args.format(
            "rzx({0:half_turns}) {1},{2};\n",
            self.exponent,
            qubits[0],
            qubits[1],
        )

    def __str__(self) -> str:
        if self.exponent == 1:
            return "ZX"
        return f"ZX**{self._exponent!r}"

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return "css.ZX"
            return f"(css.ZX**{cirq._compat.proper_repr(self._exponent)})"
        return (
            f"css.ZXPowGate(exponent={cirq._compat.proper_repr(self._exponent)},"
            f" global_shift={self._global_shift!r})"
        )


@cirq.value_equality(approximate=True)
class AceCR(cirq.Gate):
    """Active Cancellation Echoed Cross Resonance gate, supporting polarity switches and sandwiches.

    The typical AceCR in literature is a positive half-CR, then X on "Z side", then negative
    half-CR ("Z side" and "X side" refer to the two sides of the underlying ZX interactions).

    Args:
        polarity: Should be either "+-" or "-+". Specifies if positive or negative half-CR is first
        sandwich_rx_rads: Angle of rotation for an rx gate applied to the "X side" simultaneously
            with the X gate on the "Z side".
    """

    def __init__(self, polarity: str, sandwich_rx_rads: cirq.TParamVal = 0) -> None:
        if polarity not in ("+-", "-+"):
            raise ValueError("Polarity must be either '+-' or '-+'")
        self.polarity = polarity
        self.sandwich_rx_rads = sandwich_rx_rads

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(
        self, qubits: Tuple[cirq.LineQubit, cirq.LineQubit]
    ) -> Iterator[cirq.Operation]:
        yield css.CR(*qubits) ** 0.25 if self.polarity == "+-" else css.CR(*qubits) ** -0.25
        yield cirq.X(qubits[0])
        if self.sandwich_rx_rads:
            yield cirq.rx(self.sandwich_rx_rads)(qubits[1])
        yield css.CR(*qubits) ** -0.25 if self.polarity == "+-" else css.CR(*qubits) ** 0.25

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        top, bottom = f"AceCR{self.polarity}(Z side)", f"AceCR{self.polarity}(X side)"
        if self.sandwich_rx_rads:
            bottom += f"|Rx({args.format_radians(self.sandwich_rx_rads)})|"
        return cirq.CircuitDiagramInfo(wire_symbols=(top, bottom))

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, cirq.Qid]) -> Optional[str]:
        """QASM symbol for AceCR("+-") (AceCR("-+")) is acecr_pm (acecr_mp)

        If there is a sandwich, it comes last. For example, AceCR("-+", np.pi / 2) has qasm
        acecr_mp_rx(pi*0.5).
        """
        polarity_str = self.polarity.replace("+", "p").replace("-", "m")
        if not self.sandwich_rx_rads:
            return args.format("acecr_{} {},{};\n", polarity_str, *qubits)
        exponent = self.sandwich_rx_rads / np.pi
        return args.format("acecr_{}_rx({:half_turns}) {},{};\n", polarity_str, exponent, *qubits)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.sandwich_rx_rads)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.sandwich_rx_rads)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> "AceCR":
        return AceCR(
            self.polarity, cirq.resolve_parameters(self.sandwich_rx_rads, resolver, recursive)
        )

    def _value_equality_values_(self) -> Tuple[str, cirq.TParamVal]:
        if cirq.is_parameterized(self.sandwich_rx_rads):
            return self.polarity, self.sandwich_rx_rads

        return self.polarity, self.sandwich_rx_rads % (4 * np.pi)

    def _value_equality_approximate_values_(self) -> Tuple[str, cirq.PeriodicValue]:
        return self.polarity, cirq.PeriodicValue(self.sandwich_rx_rads, 4 * np.pi)

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        if not isinstance(other, AceCR):
            return NotImplemented

        if other.polarity != self.polarity:
            return False

        return approx_eq_mod(self.sandwich_rx_rads, other.sandwich_rx_rads, 2 * np.pi, atol=atol)

    def __repr__(self) -> str:
        if not self.sandwich_rx_rads:
            return f"css.AceCR({self.polarity!r})"
        return f"css.AceCR({self.polarity!r}, {self.sandwich_rx_rads!r})"

    def __str__(self) -> str:
        if not self.sandwich_rx_rads:
            return f"AceCR{self.polarity}"
        return f"AceCR{self.polarity}|{cirq.rx(self.sandwich_rx_rads)}|"

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["polarity", "sandwich_rx_rads"])


AceCRMinusPlus = AceCR("-+")

AceCRPlusMinus = AceCR("+-")


class Barrier(cirq.ops.IdentityGate, cirq.InterchangeableQubitsGate):
    """Barrier: temporal boundary restricting circuit compilation and pulse scheduling.
    Otherwise equivalent to the identity gate.
    """

    def _decompose_(self, qubits: Tuple[cirq.Qid, ...]) -> cirq.type_workarounds.NotImplementedType:
        return NotImplemented

    def _trace_distance_bound_(self) -> float:
        return 1.0

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> str:
        indices_str = ",".join([f"{{{i}}}" for i in range(len(qubits))])
        format_str = f"barrier {indices_str};\n"
        return args.format(format_str, *qubits)

    def __str__(self) -> str:
        if all(d == 2 for d in self._qid_shape):
            return f"Barrier({self.num_qubits()})"
        return f"Barrier(qid_shape={cirq.qid_shape(self)})"

    def __repr__(self) -> str:
        return f"css.{self}"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        if args.use_unicode_characters:
            return ("│",) * self.num_qubits()
        return ("|",) * self.num_qubits()


def barrier(*qubits: cirq.Qid) -> cirq.Operation:
    qid_shape = tuple(q.dimension for q in qubits)
    return css.Barrier(qid_shape=qid_shape).on(*qubits)


@cirq.value_equality(approximate=True)
class ParallelGates(cirq.Gate, cirq.InterchangeableQubitsGate):
    """A single Gate combining a collection of concurrent Gate(s) acting on different qubits.

    WARNING: for cirq versions 0.14.*, equality check will return False after permutations of
        qubits between identical but nonadjacent gates, e.g.::

            gate = ParallelGates(cirq.X, cirq.Y, cirq.X)
            gate.on(q0, q1, q2) == gate.on(q2, q1, q0)  # True for cirq < 0.14.0
                                                        # False for 0.14.0 <= cirq < 0.15.0
                                                        # True for cirq >= 0.15.0

        This does not affect permutations of qubits between adjacent gates, or those within the
        same InterchangeableQubitsGate::

            gate = ParallelGates(cirq.X, cirq.X, cirq.CZ)
            gate.on(q0, q1, q2, q3) == gate.on(q1, q0, q3, q2)  # always True

        See https://github.com/quantumlib/Cirq/issues/5148 for more information.
    """

    def __init__(self, *component_gates: cirq.Gate) -> None:
        """
        Args:
            component_gates: Gate(s) to be collected into single gate
        """

        self.component_gates: Tuple[cirq.Gate, ...] = ()

        # unroll any ParallelGate(s) instances in component_gates
        for gate in component_gates:
            if not isinstance(gate, cirq.Gate):
                raise ValueError(f"{gate} is not a cirq Gate")
            elif cirq.is_measurement(gate):
                raise ValueError("ParallelGates cannot contain measurements")
            elif isinstance(gate, ParallelGates):
                self.component_gates += gate.component_gates
            elif isinstance(gate, cirq.ParallelGate):
                self.component_gates += gate.num_copies * (gate.sub_gate,)
            else:
                self.component_gates += (gate,)

    def qubit_index_to_gate_and_index(self, index: int) -> Tuple[cirq.Gate, int]:
        for gate in self.component_gates:
            if gate.num_qubits() > index >= 0:
                return gate, index
            index -= gate.num_qubits()
        raise ValueError("index out of range")

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        indexed_gate, index_in_gate = self.qubit_index_to_gate_and_index(index)
        if indexed_gate.num_qubits() == 1:
            # find the first instance of the same gate
            first_instance = self.component_gates.index(indexed_gate)
            return sum(map(cirq.num_qubits, self.component_gates[:first_instance]))
        if isinstance(indexed_gate, cirq.InterchangeableQubitsGate):
            gate_key = indexed_gate.qubit_index_to_equivalence_group_key(index_in_gate)
            for i in range(index_in_gate):
                if gate_key == indexed_gate.qubit_index_to_equivalence_group_key(i):
                    return index - index_in_gate + i
        return index

    def _is_parameterized_(self) -> bool:
        return any(cirq.is_parameterized(gate) for gate in self.component_gates)

    def _parameter_names_(self) -> Set[str]:
        component_param_names = [set(cirq.parameter_names(gate)) for gate in self.component_gates]
        return set.union(*component_param_names)

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> "ParallelGates":
        return ParallelGates(
            *(cirq.resolve_parameters(gate, resolver, recursive) for gate in self.component_gates)
        )

    def _value_equality_values_(self) -> Tuple[cirq.Gate, ...]:
        return self.component_gates

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        if not isinstance(other, ParallelGates):
            return NotImplemented

        if len(other.component_gates) != len(self.component_gates):
            return False

        if other.num_qubits() != self.num_qubits():
            return False

        return all(
            cirq.equal_up_to_global_phase(gate, other_gate, atol=atol)
            for gate, other_gate in zip(self.component_gates, other.component_gates)
        )

    def _qid_shape_(self) -> Tuple[int, ...]:
        qid_shape: Tuple[int, ...] = ()
        for gate in self.component_gates:
            qid_shape += cirq.qid_shape(gate)
        return qid_shape

    def _decompose_(self, qubits: Tuple[cirq.Qid, ...]) -> Iterator[cirq.Operation]:
        """Decompose into each component gate"""
        for gate in self.component_gates:
            num_qubits = gate.num_qubits()
            yield gate(*qubits[:num_qubits])
            qubits = qubits[num_qubits:]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Generate a circuit diagram by connecting the wire symbols of each component gate.

        Symbols belonging to separate gates are differentiated via subscripts, with groups of
        symbols sharing the same subscript indicating multi-qubit operations.
        """
        wire_symbols_with_subscripts: List[str] = []
        for i, gate in enumerate(self.component_gates):
            qubit_index = len(wire_symbols_with_subscripts)
            num_qubits = gate.num_qubits()
            sub_args = cirq.CircuitDiagramInfoArgs(
                known_qubit_count=(num_qubits if args.known_qubit_count is not None else None),
                known_qubits=(
                    args.known_qubits[qubit_index:][:num_qubits]
                    if args.known_qubits is not None
                    else None
                ),
                use_unicode_characters=args.use_unicode_characters,
                precision=args.precision,
                label_map=args.label_map,
            )

            sub_info = cirq.circuit_diagram_info(gate, sub_args, None)
            if sub_info is None:
                return NotImplemented

            full_wire_symbols = sub_info._wire_symbols_including_formatted_exponent(
                sub_args, preferred_exponent_index=num_qubits - 1
            )

            index_str = f"_{i + 1}"
            if args.use_unicode_characters:
                index_str = "".join(chr(ord("₁") + int(c)) for c in str(i))

            for base_symbol, full_symbol in zip(sub_info.wire_symbols, full_wire_symbols):
                wire_symbols_with_subscripts.append(
                    full_symbol.replace(base_symbol, base_symbol + index_str)
                )

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols_with_subscripts)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["component_gates"])

    @classmethod
    def _from_json_dict_(cls, component_gates: List[cirq.Gate], **kwargs: Any) -> "ParallelGates":
        return cls(*component_gates)

    def __pow__(self, exponent: cirq.TParamVal) -> "ParallelGates":
        exponentiated_gates = [gate**exponent for gate in self.component_gates]
        return ParallelGates(*exponentiated_gates)

    def __str__(self) -> str:
        component_gates_str = ", ".join(str(gate) for gate in self.component_gates)
        return f"ParallelGates({component_gates_str})"

    def __repr__(self) -> str:
        component_gates_repr = ", ".join(repr(gate) for gate in self.component_gates)
        return f"css.ParallelGates({component_gates_repr})"


def parallel_gates_operation(*ops: cirq.Operation) -> cirq.Operation:
    """Given operations acting on disjoint qubits, constructs a single css.ParallelGates instance
    and applies it such that each operation's .gate is applied to its .qubits.

    Args:
        ops: operations to pack into a single ParallelGates operation

    Returns:
        ParallelGates(op.gate, op2.gate, ...).on(*op.qubits, *op2.qubits, ...)
    """
    gates: List[cirq.Gate] = []
    qubits: List[cirq.Qid] = []
    for op in ops:
        if not op.gate:
            raise ValueError(f"Invalid operation: {op} has no .gate attribute")
        if op.tags:
            raise ValueError("Invalid operation: tagged operations not permitted")

        gates.append(op.gate)
        qubits.extend(op.qubits)

    return ParallelGates(*gates).on(*qubits)


class RGate(cirq.PhasedXPowGate):
    """A single-qubit gate that rotates about an axis in the X-Y plane."""

    def __init__(self, theta: cirq.TParamVal, phi: cirq.TParamVal) -> None:
        """
        Args:
            phi: Angle (in radians) defining the axis of rotation in the `X`-`Y` plane:
                 `cos(phi) X + sin(phi) Y` (i.e. `phi` radians from `X` to `Y`).

            theta: Angle (in radians) by which to rotate.
        """
        super().__init__(
            exponent=theta / _pi(theta), phase_exponent=phi / _pi(phi), global_shift=-0.5
        )

    @property
    def phi(self) -> cirq.TParamVal:
        return self.phase_exponent * _pi(self.phase_exponent)

    @property
    def theta(self) -> cirq.TParamVal:
        return self.exponent * _pi(self.exponent)

    def __pow__(self, power: cirq.TParamVal) -> "RGate":
        return RGate(power * self.theta, self.phi)

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        """Implemented here because it isn't in cirq.PhasedXPowGate"""
        if not isinstance(other, cirq.PhasedXPowGate):
            return NotImplemented

        if approx_eq_mod(self.exponent, other.exponent, 2, atol=atol):
            return approx_eq_mod(self.phase_exponent, other.phase_exponent, 2, atol=atol)

        # equal_up_to_global_phase(RGate(x, y), RGate(-x, y + pi)) is also True:
        if approx_eq_mod(self.exponent, -other.exponent, 2, atol=atol):
            return approx_eq_mod(self.phase_exponent, 1 + other.phase_exponent, 2, atol=atol)

        return False

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        theta_str = args.format_radians(self.theta)
        phi_str = args.format_radians(self.phi)
        gate_str = f"RGate({theta_str}, {phi_str})"
        return cirq.CircuitDiagramInfo(wire_symbols=(gate_str,))

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> Optional[str]:
        return args.format(
            "r({0:half_turns},{1:half_turns}) {2};\n",
            self.exponent,
            self.phase_exponent,
            qubits[0],
        )

    def __str__(self) -> str:
        return f"RGate({self.exponent}π, {self.phase_exponent}π)"

    def __repr__(self) -> str:
        return f"css.RGate({self.theta}, {self.phi})"

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["theta", "phi"])


@cirq.value_equality(approximate=True)
class ParallelRGate(cirq.ParallelGate, cirq.InterchangeableQubitsGate):
    """Wrapper class to define a ParallelGate of identical RGate gates."""

    def __init__(self, theta: cirq.TParamVal, phi: cirq.TParamVal, num_copies: int) -> None:
        super().__init__(css.RGate(theta, phi), num_copies)
        self._sub_gate: RGate

    @property
    def sub_gate(self) -> RGate:
        return self._sub_gate

    @property
    def phase_exponent(self) -> cirq.TParamVal:
        return self.sub_gate.phase_exponent

    @property
    def exponent(self) -> cirq.TParamVal:
        return self.sub_gate.exponent

    @property
    def phi(self) -> cirq.TParamVal:
        return self.sub_gate.phi

    @property
    def theta(self) -> cirq.TParamVal:
        return self.sub_gate.theta

    def __pow__(self, power: cirq.TParamVal) -> "ParallelRGate":
        return ParallelRGate(power * self.theta, self.phi, self.num_copies)

    def _equal_up_to_global_phase_(self, other: Any, atol: float) -> Optional[bool]:
        """Implemented here because it isn't in cirq.ParallelGate"""
        if not isinstance(other, cirq.ParallelGate):
            return NotImplemented

        if self.num_copies != other.num_copies:
            return False

        return cirq.equal_up_to_global_phase(self.sub_gate, other.sub_gate)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        diagram_info = cirq.circuit_diagram_info(self.sub_gate, args)
        wire_symbols = tuple(diagram_info.wire_symbols) + tuple(
            f"#{idx}" for idx in range(2, self.num_copies + 1)
        )
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> str:
        gate_str = "gate_GR({0:half_turns},{1:half_turns})"
        qubits_str = ",".join([f"{{{idx + 2}}}" for idx in range(len(qubits))])
        return args.format(
            f"{gate_str} {qubits_str};\n", self.exponent, self.phase_exponent, *qubits
        )

    def __str__(self) -> str:
        return f"RGate({self.phase_exponent}π, {self.exponent}π) x {self.num_copies}"

    def __repr__(self) -> str:
        return f"css.ParallelRGate({self.theta}, {self.phi}, {self.num_copies})"

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["theta", "phi", "num_copies"])


class IXGate(cirq.XPowGate):
    """Thin wrapper of Rx(-pi) to improve iToffoli circuit diagrams"""

    def __init__(self) -> None:
        super().__init__(exponent=1, global_shift=0.5)

    def _with_exponent(self, exponent: cirq.value.TParamVal) -> Union[cirq.Rx, "IXGate"]:
        if approx_eq_mod(exponent, 1.0, 4):
            return IXGate()
        return cirq.rx(-exponent * _pi(exponent))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=("iX",))

    def __str__(self) -> str:
        return "IX"

    def __repr__(self) -> str:
        return f"css.ops.qubit_gates.{str(self)}"

    @classmethod
    def _from_json_dict_(cls, **kwargs: Any) -> "IXGate":
        return IXGate()


CR = ZX = ZXPowGate()  # standard CR is a full turn of ZX, i.e. exponent = 1

IX = IXGate()

# iToffoli gate
ICCX = IX.controlled(2, [1, 1])

# Open-control iToffoli gate
AQTICCX = AQTITOFFOLI = IX.controlled(2, [0, 0])


def custom_resolver(cirq_type: str) -> Union[Type[cirq.Gate], None]:
    if cirq_type == "ZZSwapGate":
        return ZZSwapGate
    if cirq_type == "Barrier":
        return Barrier
    if cirq_type == "ZXPowGate":
        return ZXPowGate
    if cirq_type == "AceCR":
        return AceCR
    if cirq_type == "ParallelGates":
        return ParallelGates
    if cirq_type == "MSGate":
        return cirq.ops.MSGate
    if cirq_type == "RGate":
        return RGate
    if cirq_type == "IXGate":
        return IXGate
    if cirq_type == "ParallelRGate":
        return ParallelRGate

    return None
