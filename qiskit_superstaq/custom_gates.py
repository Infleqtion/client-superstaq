import functools
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import qiskit


class AceCR(qiskit.circuit.Gate):
    """Active Cancellation Echoed Cross Resonance gate, supporting polarity switches and sandwiches.

    The typical AceCR in literature is a positive half-CR, then X on "Z side", then negative
    half-CR ("Z side" and "X side" refer to the two sides of the underlying ZX interactions).
    Args:
        polarity: Should be either "+-" or "-+". Specifies if positive or negative half-CR is first
        sandwich_rx_rads: Angle of rotation for an rx gate applied to the "X side" simultaneously
            with the X gate on the "Z side".
        label: An optional label for the constructed Gate
    """

    def __init__(
        self, polarity: str, sandwich_rx_rads: float = 0, label: Optional[str] = None
    ) -> None:
        if polarity not in ("+-", "-+"):
            raise ValueError("Polarity must be either '+-' or '-+'")

        name = "acecr_" + polarity.replace("+", "p").replace("-", "m")

        if sandwich_rx_rads:
            super().__init__(name + "_rx", 2, [sandwich_rx_rads], label=label)
        else:
            super().__init__(name, 2, [], label=label)

        self.polarity = polarity
        self.sandwich_rx_rads = sandwich_rx_rads

    def inverse(self) -> "AceCR":
        return AceCR(self.polarity, sandwich_rx_rads=-self.sandwich_rx_rads, label=self.label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name=self.name)
        first_sign = +1 if self.polarity == "+-" else -1
        qc.rzx(first_sign * np.pi / 4, 0, 1)
        qc.x(0)
        if self.sandwich_rx_rads:
            qc.rx(self.sandwich_rx_rads, 1)
        qc.rzx(-first_sign * np.pi / 4, 0, 1)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
        cval = 1 / np.sqrt(2)
        if self.polarity == "+-":
            sval = 1j * cval
        else:
            sval = -1j * cval

        mat = np.array(
            [
                [0, cval, 0, sval],
                [cval, 0, -sval, 0],
                [0, sval, 0, cval],
                [-sval, 0, cval, 0],
            ],
            dtype=dtype,
        )

        # sandwiched rx gate commutes and can just be multiplied with non-sandwiched part:
        return mat @ np.kron(
            np.asarray(qiskit.circuit.library.RXGate(self.sandwich_rx_rads), dtype=dtype),
            np.eye(2, dtype=dtype),
        )

    def __repr__(self) -> str:
        args = f"'{self.polarity}'"
        if self.sandwich_rx_rads:
            args += f", sandwich_rx_rads={self.sandwich_rx_rads}"
        if self.label:
            args += f", label='{self.label}'"
        return f"qss.AceCR({args})"

    def __str__(self) -> str:
        if not self.sandwich_rx_rads:
            return f"AceCR{self.polarity}"
        arg = qiskit.circuit.tools.pi_check(self.sandwich_rx_rads, ndigits=8, output="qasm")
        return f"AceCR{self.polarity}|RXGate({arg})|"


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
        """
        Args:
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

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
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


class ParallelGates(qiskit.circuit.Gate):
    """A single Gate combining a collection of concurrent Gate(s) acting on different qubits"""

    def __init__(self, *component_gates: qiskit.circuit.Gate, label: Optional[str] = None) -> None:
        """
        Args:
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

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
        mat = functools.reduce(np.kron, (gate.to_matrix() for gate in self.component_gates[::-1]))
        return np.asarray(mat, dtype=dtype)

    def __str__(self) -> str:
        args = ", ".join(gate.qasm() for gate in self.component_gates)
        return f"ParallelGates({args})"


class iXGate(qiskit.circuit.Gate):
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__("ix", 1, [], label=label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(1, name=self.name)
        qc.rx(-np.pi, 0)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
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


class iXdgGate(qiskit.circuit.Gate):
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__("ixdg", 1, [], label=label)

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(1, name=self.name)
        qc.rx(np.pi, 0)
        self.definition = qc

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
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


class iCCXGate(qiskit.circuit.ControlledGate):
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

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
        mat = qiskit.circuit._utils._compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        return np.asarray(mat, dtype=dtype)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iCCXGate(label={self.label}, ctrl_state={self.ctrl_state})"


class iCCXdgGate(qiskit.circuit.ControlledGate):
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

    def __array__(self, dtype: Optional[type] = None) -> np.ndarray:
        mat = qiskit.circuit._utils._compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        return np.asarray(mat, dtype=dtype)

    def __repr__(self) -> str:
        return f"qss.custom_gates.{str(self)}"

    def __str__(self) -> str:
        return f"iCCXdgGate(label={self.label}, ctrl_state={self.ctrl_state})"


class AQTiCCXGate(iCCXGate):
    def __init__(self, label: Optional[str] = None) -> None:
        super().__init__(label=label, ctrl_state="00")


AQTiToffoliGate = AQTiCCXGate


_custom_gate_resolvers: Dict[str, Callable[..., qiskit.circuit.Gate]] = {
    "acecr_pm": lambda: AceCR("+-"),
    "acecr_mp": lambda: AceCR("-+"),
    "acecr_pm_rx": lambda rads: AceCR("+-", rads),
    "acecr_mp_rx": lambda rads: AceCR("-+", rads),
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
