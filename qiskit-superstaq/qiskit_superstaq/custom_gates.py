import functools
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import qiskit


class AceCR(qiskit.circuit.Gate):
    """Active Cancellation Echoed Cross Resonance gate, supporting polarity switches and sandwiches.

    The typical AceCR in literature is a positive half-CR, then X on "Z side", then negative
    half-CR ("Z side" and "X side" refer to the two sides of the underlying ZX interactions).

    Args:
        rads: Angle of rotation for CR gate (i.e., twice the angle for each echoed half-CR).
        sandwich_rx_rads: Angle of rotation for an rx gate applied to the "X side" simultaneously
            with the X gate on the "Z side".
        label: An optional label for the constructed Gate.
    """

    def __init__(
        self,
        rads: Union[str, float] = np.pi / 2,
        sandwich_rx_rads: float = 0,
        label: Optional[str] = None,
    ) -> None:
        if rads == "+-":
            rads = np.pi / 2
        elif rads == "-+":
            rads = -np.pi / 2
        elif isinstance(rads, str):
            raise ValueError("Polarity must be either '+-' or '-+'")

        name = "acecr"
        params = [rads]
        if sandwich_rx_rads:
            name += "_rx"
            params.append(sandwich_rx_rads)
        super().__init__(name, 2, params, label=label)

        self.rads = rads
        self.sandwich_rx_rads = sandwich_rx_rads

    def inverse(self) -> "AceCR":
        return AceCR(self.rads, -self.sandwich_rx_rads, label=self.label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name=self.name)
        qc.rzx(self.rads / 2, 0, 1)
        qc.x(0)
        if self.sandwich_rx_rads:
            qc.rx(self.sandwich_rx_rads, 1)
        qc.rzx(-self.rads / 2, 0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.bool_]:
        return qiskit.quantum_info.Operator(self.definition).data

    def __repr__(self) -> str:
        args = []
        if self.rads != np.pi / 2:
            args.append(f"rads={self.rads!r}")
        if self.sandwich_rx_rads != 0:
            args.append(f"sandwich_rx_rads={self.sandwich_rx_rads!r}")
        if self.label:
            args.append(f"label={self.label!r}")
        return f"qss.AceCR({', '.join(args)})"

    def __str__(self) -> str:
        rads_str = f"{self.rads}"
        if np.isclose(round(self.rads / np.pi, 5) * np.pi, self.rads):
            rads_str = f"{round(self.rads / np.pi, 5)}π"
        if self.sandwich_rx_rads == 0 and self.rads == np.pi / 2:
            return "AceCR"
        elif self.sandwich_rx_rads != 0 and self.rads not in [0, np.pi / 2]:
            arg = qiskit.circuit.tools.pi_check(self.sandwich_rx_rads, ndigits=8, output="qasm")
            return f"AceCR({rads_str})|RXGate({arg})|"
        elif self.sandwich_rx_rads != 0:
            arg = qiskit.circuit.tools.pi_check(self.sandwich_rx_rads, ndigits=8, output="qasm")
            return f"AceCR|RXGate({arg})|"
        else:
            return f"AceCR({rads_str})"


class ZZSwapGate(qiskit.circuit.Gate):
    r"""The ZZ-SWAP gate, which performs the ZZ-interaction followed by a SWAP.

    ZZ-SWAPs are useful for applications like QAOA or Hamiltonian Simulation,
    particularly on linear- or low- connectivity devices. See https://arxiv.org/pdf/2004.14970.pdf
    for an application of ZZ-SWAP networks.

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

    def __init__(self, theta: float, label: Optional[str] = None) -> None:
        """Args:
        theta: ZZ-interaction angle in radians
        label: an optional label for the constructed Gate
        """
        super().__init__("zzswap", 2, [theta], label=label)

    def inverse(self) -> "ZZSwapGate":
        return ZZSwapGate(-self.params[0])

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name="zzswap")
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.p(self.params[0], 1)
        qc.cx(0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, np.exp(1j * self.params[0]), 0],
                [0, np.exp(1j * self.params[0]), 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    def __repr__(self) -> str:
        args = f"{self.params[0]}"
        if self.label:
            args += f", label='{self.label}'"
        return f"qss.ZZSwapGate({args})"

    def __str__(self) -> str:
        args = qiskit.circuit.tools.pi_check(self.params[0], ndigits=8, output="qasm")
        return f"ZZSwapGate({args})"


class StrippedCZGate(qiskit.circuit.Gate):
    """The Stripped CZ Gate is the gate that is actually being performed by Hilbert,
    which is just a regular CZ gate when the rz angle = 0, and is corrected into a CZ
    gate by RZ gates afterwards if the rz angle is nonzero.
    """

    def __init__(self, rz_rads: float) -> None:
        """Args: rz_rads: RZ-rotation angle in radians"""
        super().__init__("stripped_cz", 2, [rz_rads])

    def inverse(self) -> "StrippedCZGate":
        return StrippedCZGate(-self.params[0])

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name="stripped_cz")
        qc.p(self.params[0], 0)
        qc.p(self.params[0], 1)
        qc.cz(0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.complex_]:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.exp(1j * self.params[0]), 0, 0],
                [0, 0, np.exp(1j * self.params[0]), 0],
                [0, 0, 0, np.exp(1j * (2 * self.params[0] - np.pi))],
            ],
            dtype=dtype,
        )

    def __repr__(self) -> str:
        return f"qss.StrippedCZGate({self.params[0]!r})"

    def __str__(self) -> str:
        args = qiskit.circuit.tools.pi_check(self.params[0], ndigits=8, output="qasm")
        return f"StrippedCZGate({args})"


class ParallelGates(qiskit.circuit.Gate):
    """A single Gate combining a collection of concurrent Gate(s) acting on different qubits"""

    def __init__(self, *component_gates: qiskit.circuit.Gate, label: Optional[str] = None) -> None:
        """Args:
        component_gates: Gate(s) to be collected into single gate
        label: an optional label for the constructed Gate
        """
        self.component_gates: Tuple[qiskit.circuit.Gate, ...] = ()
        num_qubits = 0

        for gate in component_gates:
            num_qubits += gate.num_qubits

            if not isinstance(gate, qiskit.circuit.Gate):
                raise ValueError("Component gates must be instances of qiskit.circuit.Gate")
            elif isinstance(gate, ParallelGates):
                self.component_gates += gate.component_gates
            else:
                self.component_gates += (gate,)

        name = "parallel_" + "_".join(gate.name for gate in self.component_gates)
        super().__init__(name, num_qubits, [], label=label)

    def inverse(self) -> "ParallelGates":
        return ParallelGates(*[gate.inverse() for gate in self.component_gates])

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(self.num_qubits, name="parallel_gates")
        qubits = list(range(self.num_qubits))
        for gate in self.component_gates:
            num_qubits = gate.num_qubits
            qc.append(gate, qubits[:num_qubits])
            qubits = qubits[num_qubits:]
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.bool_]:
        mat = functools.reduce(np.kron, (gate.to_matrix() for gate in self.component_gates[::-1]))
        return np.asarray(mat, dtype=dtype)

    def __str__(self) -> str:
        args = ", ".join(gate.qasm() for gate in self.component_gates)
        return f"ParallelGates({args})"


class iXGate(qiskit.circuit.Gate):  # pylint: disable=missing-class-docstring
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__("ix", 1, [], label=label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(1, name=self.name)
        qc.rx(-np.pi, 0)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.bool_]:
        return np.array([[0, 1j], [1j, 0]])

    def inverse(self) -> "iXdgGate":
        return iXdgGate()

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ) -> qiskit.circuit.ControlledGate:
        if num_ctrl_qubits == 2:
            gate = iCCXGate(ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits, label, ctrl_state)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iXGate(label={self.label})"


class iXdgGate(qiskit.circuit.Gate):  # pylint: disable=missing-class-docstring
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__("ixdg", 1, [], label=label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(1, name=self.name)
        qc.rx(np.pi, 0)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.complex_]:
        return np.array([[0, -1j], [-1j, 0]])

    def inverse(self) -> iXGate:
        return iXGate()

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ) -> qiskit.circuit.ControlledGate:
        if num_ctrl_qubits == 2:
            gate = iCCXdgGate(ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits, label, ctrl_state)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iXdgGate(label={self.label})"


class iCCXGate(qiskit.circuit.ControlledGate):  # pylint: disable=missing-class-docstring
    def __init__(
        self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None
    ) -> None:
        super().__init__(
            "iccx", 3, [], label, num_ctrl_qubits=2, ctrl_state=ctrl_state, base_gate=iXGate()
        )

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(3, name=self.name)
        qc.ccx(0, 1, 2)
        qc.cp(np.pi / 2, 0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.complex_]:
        mat = qiskit.circuit._utils._compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        return np.asarray(mat, dtype=dtype)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iCCXGate(label={self.label}, ctrl_state={self.ctrl_state})"


class iCCXdgGate(qiskit.circuit.ControlledGate):  # pylint: disable=missing-class-docstring
    def __init__(
        self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None
    ) -> None:
        super().__init__(
            "iccxdg", 3, [], label, num_ctrl_qubits=2, ctrl_state=ctrl_state, base_gate=iXdgGate()
        )

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(3, name=self.name)
        qc.ccx(0, 1, 2).inverse()
        qc.cp(-np.pi / 2, 0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> npt.NDArray[np.complex_]:
        mat = qiskit.circuit._utils._compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        return np.asarray(mat, dtype=dtype)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iCCXdgGate(label={self.label}, ctrl_state={self.ctrl_state})"


class AQTiCCXGate(iCCXGate):  # pylint: disable=missing-class-docstring
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__(label=label, ctrl_state="00")


AQTiToffoliGate = AQTiCCXGate


_custom_gate_resolvers: Dict[str, Callable[..., qiskit.circuit.Gate]] = {
    "acecr": lambda rads: AceCR(rads=rads),
    "acecr_rx": lambda *rads: AceCR(*rads),
    "zzswap": ZZSwapGate,
    "ix": iXGate,
    "ixdg": iXdgGate,
    "iccx": iCCXGate,
    "iccx_o0": AQTiCCXGate,
    "iccx_o1": lambda: iCCXGate(ctrl_state="01"),
    "iccx_o2": lambda: iCCXGate(ctrl_state="10"),
    "iccxdg": iCCXdgGate,
    "iccxdg_o0": lambda: iCCXdgGate(ctrl_state="00"),
    "iccxdg_o1": lambda: iCCXdgGate(ctrl_state="01"),
    "iccxdg_o2": lambda: iCCXdgGate(ctrl_state="10"),
}


def custom_resolver(gate: qiskit.circuit.Instruction) -> Optional[qiskit.circuit.Gate]:
    """Recover a custom gate type from a generic qiskit.circuit.Gate. Resolution is done using
    gate.definition.name rather than gate.name, as the former is set by all qiskit-superstaq
    custom gates and the latter may be modified by calls such as QuantumCircuit.qasm()
    """
    if gate.definition and gate.definition.name == "parallel_gates":
        component_gates = [custom_resolver(inst) or inst for inst, _, _ in gate.definition]
        return ParallelGates(*component_gates, label=gate.label)

    if gate.definition and gate.definition.name in _custom_gate_resolvers:
        new_gate = _custom_gate_resolvers[gate.definition.name](*gate.params)
    elif gate.name in _custom_gate_resolvers:
        new_gate = _custom_gate_resolvers[gate.name](*gate.params)
    else:
        return None

    new_gate.label = gate.label
    return new_gate
