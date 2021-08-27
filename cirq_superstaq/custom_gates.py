"""Miscellaneous custom gates that we encounter and want to explicitly define."""

from typing import Any, Callable, Dict, Sequence, Tuple, Union

import cirq
import numpy as np


@cirq.value_equality(approximate=True)
class FermionicSWAPGate(
    cirq.ops.gate_features.TwoQubitGate, cirq.ops.gate_features.InterchangeableQubitsGate
):
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

    Note that this gate is NOT the same as ``cirq.FSimGate``.
    """

    def __init__(self, theta: float) -> None:
        """
        Args:
            theta: ZZ-interaction angle in radians
        """
        self.theta = cirq.ops.fsim_gate._canonicalize(theta)  # between -pi and +pi

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, np.exp(1j * self.theta), 0],
                [0, np.exp(1j * self.theta), 0, 0],
                [0, 0, 0, 1],
            ]
        )

    def _value_equality_values_(self) -> Any:
        return self.theta

    def __str__(self) -> str:
        return f"FermionicSWAPGate({self.theta})"

    def __repr__(self) -> str:
        return f"cirq_superstaq.custom_gates.FermionicSWAPGate({self.theta})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        t = args.format_radians(self.theta)
        return cirq.CircuitDiagramInfo(wire_symbols=(f"FermionicSWAP({t})", f"FermionicSWAP({t})"))

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ["theta"])


class Barrier(cirq.ops.IdentityGate):
    """Barrier: temporal boundary restricting circuit compilation and pulse scheduling.
    Otherwise equivalent to the identity gate.
    """

    def _decompose_(self, qubits: Sequence["cirq.Qid"]) -> cirq.type_workarounds.NotImplementedType:
        return NotImplemented

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]) -> str:
        indices_str = ",".join([f"{{{i}}}" for i in range(len(qubits))])
        format_str = f"barrier {indices_str};\n"
        return args.format(format_str, *qubits)

    def __str__(self) -> str:
        return f"Barrier({self.num_qubits()})"

    def __repr__(self) -> str:
        return f"cirq_superstaq.custom_gates.Barrier({self.num_qubits()})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        return ("|",) * self.num_qubits()


def custom_resolver(cirq_type: str) -> Union[Callable[..., cirq.Gate], None]:
    if cirq_type == "FermionicSWAPGate":
        return FermionicSWAPGate
    if cirq_type == "Barrier":
        return Barrier
    return None
