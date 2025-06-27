import math

import cirq
import sympy


def msd_5_to_1(qubits: list[cirq.LineQubit]) -> cirq.Circuit:
    """Function to perform a 5-to-1 magic state distillation protocol.
        Reference: https://arxiv.org/pdf/2310.12106 Page 4 Figure 1.

    Args:
        qubits: The list of LineQubits of length 5.
                The first qubit will be the final magic state qubit.

    Returns:
        The magic state distillation circuit.
    """
    phi = math.acos(1 / math.sqrt(3))
    theta = math.pi / 4

    cir = cirq.Circuit()

    for q in qubits:
        cir.append([cirq.R(q)])

    cir.append(
        [
            cirq.ry(phi)(qubits[0]),
            cirq.ry(phi)(qubits[1]),
            cirq.ry(phi)(qubits[2]),
            cirq.ry(phi)(qubits[3]),
            cirq.ry(phi)(qubits[4]),
        ],
    )

    cir.append(
        [
            cirq.rz(theta)(qubits[0]),
            cirq.rz(theta)(qubits[1]),
            cirq.rz(theta)(qubits[2]),
            cirq.rz(theta)(qubits[3]),
            cirq.rz(theta)(qubits[4]),
        ],
    )

    # adjoint encode
    cir.append(
        [
            cirq.X(qubits[1]),
            cirq.X(qubits[2]),
            cirq.Z(qubits[3]),
            cirq.Y(qubits[4]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.CNOT(qubits[0], qubits[3]),
            cirq.CNOT(qubits[0], qubits[4]),
        ]
    )

    cir.append([cirq.H(qubits[4]), cirq.SWAP(qubits[3], qubits[4])])

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[3]),
            cirq.H(qubits[4]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[3], qubits[4]),
            cirq.CNOT(qubits[1], qubits[4]),
            cirq.H(qubits[3]),
            cirq.CNOT(qubits[0], qubits[3]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.H(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
        ],
    )

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[4], qubits[3]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[4], qubits[2]),
            cirq.CNOT(qubits[3], qubits[2]),
            cirq.CNOT(qubits[2], qubits[1]),
            cirq.CNOT(qubits[4], qubits[0]),
            cirq.CNOT(qubits[3], qubits[0]),
            cirq.CNOT(qubits[1], qubits[0]),
        ],
    )

    cir.append(
        [
            cirq.measure(qubits[1], key="m1"),
            cirq.measure(qubits[2], key="m2"),
            cirq.measure(qubits[3], key="m3"),
            cirq.measure(qubits[4], key="m4"),
        ],
    )

    m1, m2, m3, m4 = sympy.symbols("m1 m2 m3 m4")
    sympy_cond = cirq.SympyCondition(sympy.Eq(m1 + m2 + m3 + m4, 0))  # m1 = m2 = m3 = m4 = 0

    magic_state_cir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(),
            use_repetition_ids=False,
            repeat_until=sympy_cond,
        )
    )

    return magic_state_cir
