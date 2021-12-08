import functools
from typing import Optional, Type

import numpy as np
import qiskit


class AceCR(qiskit.circuit.Gate):
    def __init__(self, polarity: str, label: Optional[str] = None) -> None:
        """
        Args:
            polarity: a str indicating the order of ZX ** ±0.25 interactions ('+-' or '-+')
            label: an optional label for the constructed Gate
        """
        if polarity not in ("+-", "-+"):
            raise ValueError("Polarity must be either '+-' or '-+'")

        name = "acecr_" + polarity.replace("+", "p").replace("-", "m")
        super().__init__(name, 2, [], label=label)
        self.polarity = polarity

    def inverse(self) -> "AceCR":
        return self.copy()

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name=self.name)
        if self.polarity == "+-":
            qc.rzx(np.pi / 4, 0, 1)
            qc.x(0)
            qc.rzx(-np.pi / 4, 0, 1)
        else:
            qc.rzx(-np.pi / 4, 0, 1)
            qc.x(0)
            qc.rzx(np.pi / 4, 0, 1)
        self.definition = qc

    def __array__(self, dtype: Type = None) -> np.ndarray:
        cval = 1 / np.sqrt(2)
        if self.polarity == "+-":
            sval = 1j * cval
        else:
            sval = -1j * cval

        return np.array(
            [
                [0, cval, 0, sval],
                [cval, 0, -sval, 0],
                [0, sval, 0, cval],
                [-sval, 0, cval, 0],
            ],
            dtype=dtype,
        )

    def __repr__(self) -> str:
        if self.label:
            return f"qiskit_superstaq.AceCR('{self.polarity}', label='{self.label}')"
        return f"qiskit_superstaq.AceCR('{self.polarity}')"

    def __str__(self) -> str:
        return f"AceCR{self.polarity}"


class FermionicSWAPGate(qiskit.circuit.Gate):
    r"""The Fermionic SWAP gate, which performs the ZZ-interaction followed by a SWAP.

    Fermionic SWAPs are useful for applications like QAOA or Hamiltonian Simulation,
    particularly on linear- or low- connectivity devices. See https://arxiv.org/pdf/2004.14970.pdf
    for an application of Fermionic SWAP networks.

    The unitary for a Fermionic SWAP gate parametrized by ZZ-interaction angle :math:`\theta` is:

     .. math::

        \begin{bmatrix}
        1 & . & . & . \\
        . & . & e^{i \theta} & . \\
        . & e^{i \theta} & . & . \\
        . & . & . & 1 \\
        \end{bmatrix}

    where '.' means '0'.
    For :math:`\theta = 0`, the Fermionic SWAP gate is just an ordinary SWAP.
    """

    def __init__(self, theta: float, label: Optional[str] = None) -> None:
        """
        Args:
            theta: ZZ-interaction angle in radians
            label: an optional label for the constructed Gate
        """
        super().__init__("fermionic_swap", 2, [theta], label=label)

    def inverse(self) -> "FermionicSWAPGate":
        return FermionicSWAPGate(-self.params[0])

    def _define(self) -> None:
        qc = qiskit.QuantumCircuit(2, name="fermionic_swap")
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.rz(self.params[0], 1)
        qc.cx(0, 1)
        self.definition = qc

    def __array__(self, dtype: Type = None) -> np.ndarray:
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
        return f"qiskit_superstaq.FermionicSWAPGate({args})"

    def __str__(self) -> str:
        args = qiskit.circuit.tools.pi_check(self.params[0], ndigits=8, output="qasm")
        return f"FermionicSWAPGate({args})"


class ParallelGates(qiskit.circuit.Gate):
    """A single Gate combining a collection of concurrent Gate(s) acting on different qubits"""

    def __init__(self, *component_gates: qiskit.circuit.Gate, label: Optional[str] = None) -> None:
        """
        Args:
            component_gates: Gate(s) to be collected into single gate
            label: an optional label for the constructed Gate
        """
        if not all(isinstance(gate, qiskit.circuit.Gate) for gate in component_gates):
            raise ValueError("Component gates must be instances of qiskit.circuit.Gate")

        num_qubits = sum(gate.num_qubits for gate in component_gates)
        name = "parallel_" + "_".join(gate.name for gate in component_gates)

        super().__init__(name, num_qubits, [], label=label)
        self.component_gates = component_gates

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

    def __array__(self, dtype: Type = None) -> np.ndarray:
        mat = functools.reduce(np.kron, (gate.to_matrix() for gate in self.component_gates[::-1]))
        return np.asarray(mat, dtype=dtype)

    def __str__(self) -> str:
        args = ", ".join(gate.qasm() for gate in self.component_gates)
        return f"ParallelGates({args})"


def custom_resolver(gate: qiskit.circuit.Gate) -> Optional[qiskit.circuit.Gate]:
    """Recover a custom gate type from a generic qiskit.circuit.Gate. Resolution is done using
    gate.definition.name rather than gate.name, as the former is set by all qiskit_superstaq
    custom gates and the latter may be modified by calls such as QuantumCircuit.qasm()
    """

    if gate.definition is None:
        return None
    if gate.definition.name == "acecr_pm":
        return AceCR("+-", label=gate.label)
    if gate.definition.name == "acecr_mp":
        return AceCR("-+", label=gate.label)
    if gate.definition.name == "fermionic_swap":
        return FermionicSWAPGate(gate.params[0], label=gate.label)
    if gate.definition.name == "parallel_gates":
        component_gates = [custom_resolver(inst) or inst for inst, _, _ in gate.definition]
        return ParallelGates(*component_gates, label=gate.label)
    return None
