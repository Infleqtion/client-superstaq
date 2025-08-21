# Copyright 2025 Infleqtion
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
from collections.abc import Sequence

import cirq
import numpy as np
import sympy


def msd_5_to_1(qubits: Sequence[cirq.Qid]) -> cirq.Circuit:
    """Function to perform a 5-to-1 magic state distillation protocol.

    Reference: https://arxiv.org/abs/2310.12106 Page 4 Figure 1.

    Args:
        qubits: A list of qubits to use in the circuit, where the first qubit will
            be the final magic state qubit.

    Returns:
        The magic state distillation circuit.
    """
    # phi is the rotation angle for preparing the |H> magic state, derived from
    # cos(phi) = 1/sqrt(3), as described in the 5-to-1 magic state distillation protocol
    # (see https://arxiv.org/abs/2310.12106, Page 4, Figure 1).
    phi = np.arccos(1 / np.sqrt(3))
    # theta is pi/4, corresponding to the T gate (pi/4 phase), as required by the protocol.

    theta = np.pi / 4

    cir = cirq.Circuit()

    for q in qubits:
        cir.append([cirq.reset(q)])

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

    cir.append(
        [
            cirq.X(qubits[0]),
            cirq.S(qubits[0]) ** -1,
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


def msd_7_to_1(qubits: Sequence[cirq.Qid]) -> cirq.Circuit:
    """Function to perform a 7-to-1 magic state distillation protocol.
        Reference: https://arxiv.org/abs/0803.0272 Page 12 Figure 19.

    Args:
        qubits: The list of `cirq.LineQubit` of length 8. The first qubit
            in the list will be the final magic state qubit.

    Returns:
        The magic state distillation circuit.
    """
    cir = cirq.Circuit()

    for q in qubits:
        cir.append([cirq.reset(q)])

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[4]),
        ],
    )

    cir.append([cirq.CNOT(qubits[0], qubits[3])])

    cir.append(
        [
            cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[4], qubits[5]),
            cirq.CNOT(qubits[4], qubits[6]),
            cirq.CNOT(qubits[4], qubits[7]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[2], qubits[3]),
            cirq.CNOT(qubits[2], qubits[6]),
            cirq.CNOT(qubits[2], qubits[7]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[1], qubits[3]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[7]),
        ],
    )

    cir.append(
        [
            cirq.S(qubits[1]),
            cirq.S(qubits[2]),
            cirq.S(qubits[3]),
            cirq.S(qubits[4]),
            cirq.S(qubits[5]),
            cirq.S(qubits[6]),
            cirq.S(qubits[7]),
        ],
    )

    # no need to measure index 0, that is our magic state
    cir.append(
        [
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[3]),
            cirq.H(qubits[4]),
            cirq.H(qubits[5]),
            cirq.H(qubits[6]),
            cirq.H(qubits[7]),
            cirq.measure(qubits[1], key="m6"),
            cirq.measure(qubits[2], key="m5"),
            cirq.measure(qubits[3], key="m4"),
            cirq.measure(qubits[4], key="m3"),
            cirq.measure(qubits[5], key="m2"),
            cirq.measure(qubits[6], key="m1"),
            cirq.measure(qubits[7], key="m0"),
        ],
    )

    # index 0=magic
    m0, m1, m2, m3, m4, m5, m6 = sympy.symbols("m0 m1 m2 m3 m4 m5 m6")
    # all those parities must be 0 for it to be even ( so output of xor is 0 for even )
    # so stop when true == ~0 & ~0 & ~0 == ~(0 | 0 | 0)
    even_parity = sympy.Not(
        sympy.Xor(m0, m1, m2, m3) | sympy.Xor(m0, m1, m4, m5) | sympy.Xor(m0, m2, m4, m6)
    )
    # if special parity is even, do Z
    special_parity = sympy.Xor(m4, m5, m6)

    # repeating until matched parities
    magic_state_cir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(),
            use_repetition_ids=False,
            repeat_until=cirq.SympyCondition(even_parity),
        )
    )

    # magic state "correction"
    # is even, so do the Z
    magic_state_cir.append([cirq.Z(qubits[0]).with_classical_controls(sympy.Not(special_parity))])

    return magic_state_cir


def msd_15_to_1(qubits: Sequence[cirq.Qid]) -> cirq.Circuit:
    """Function to perform a 15-to-1 magic state distillation protocol.
        Reference: https://arxiv.org/abs/1208.0928 Page 38 Figure 33.

    Args:
        qubits: The list of LineQubits of length 16.
                The last qubit will be the final magic state qubit.

    Returns:
        The magic state distillation circuit.
    """
    cir = cirq.Circuit()

    for q in qubits:
        cir.append([cirq.reset(q)])

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[3]),
            cirq.H(qubits[7]),
            cirq.H(qubits[15]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[15], qubits[14]),
            cirq.CNOT(qubits[7], qubits[8]),
            cirq.CNOT(qubits[7], qubits[9]),
            cirq.CNOT(qubits[7], qubits[10]),
            cirq.CNOT(qubits[7], qubits[11]),
            cirq.CNOT(qubits[7], qubits[12]),
            cirq.CNOT(qubits[7], qubits[13]),
            cirq.CNOT(qubits[7], qubits[14]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[3], qubits[4]),
            cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6]),
            cirq.CNOT(qubits[3], qubits[11]),
            cirq.CNOT(qubits[3], qubits[12]),
            cirq.CNOT(qubits[3], qubits[13]),
            cirq.CNOT(qubits[3], qubits[14]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[6]),
            cirq.CNOT(qubits[1], qubits[9]),
            cirq.CNOT(qubits[1], qubits[10]),
            cirq.CNOT(qubits[1], qubits[13]),
            cirq.CNOT(qubits[1], qubits[14]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.CNOT(qubits[0], qubits[4]),
            cirq.CNOT(qubits[0], qubits[6]),
            cirq.CNOT(qubits[0], qubits[8]),
            cirq.CNOT(qubits[0], qubits[10]),
            cirq.CNOT(qubits[0], qubits[12]),
            cirq.CNOT(qubits[0], qubits[14]),
        ],
    )

    cir.append(
        [
            cirq.CNOT(qubits[14], qubits[2]),
            cirq.CNOT(qubits[14], qubits[4]),
            cirq.CNOT(qubits[14], qubits[5]),
            cirq.CNOT(qubits[14], qubits[8]),
            cirq.CNOT(qubits[14], qubits[9]),
            cirq.CNOT(qubits[14], qubits[11]),
        ],
    )

    cir.append(
        [
            cirq.inverse(cirq.T(qubits[0])),
            cirq.inverse(cirq.T(qubits[1])),
            cirq.inverse(cirq.T(qubits[2])),
            cirq.inverse(cirq.T(qubits[3])),
            cirq.inverse(cirq.T(qubits[4])),
            cirq.inverse(cirq.T(qubits[5])),
            cirq.inverse(cirq.T(qubits[6])),
            cirq.inverse(cirq.T(qubits[7])),
            cirq.inverse(cirq.T(qubits[8])),
            cirq.inverse(cirq.T(qubits[9])),
            cirq.inverse(cirq.T(qubits[10])),
            cirq.inverse(cirq.T(qubits[11])),
            cirq.inverse(cirq.T(qubits[12])),
            cirq.inverse(cirq.T(qubits[13])),
            cirq.inverse(cirq.T(qubits[14])),
        ],
    )

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[3]),
            cirq.H(qubits[4]),
            cirq.H(qubits[5]),
            cirq.H(qubits[6]),
            cirq.H(qubits[7]),
            cirq.H(qubits[8]),
            cirq.H(qubits[9]),
            cirq.H(qubits[10]),
            cirq.H(qubits[11]),
            cirq.H(qubits[12]),
            cirq.H(qubits[13]),
            cirq.H(qubits[14]),
            cirq.measure(qubits[0], key="m1"),
            cirq.measure(qubits[1], key="m2"),
            cirq.measure(qubits[2], key="m3"),
            cirq.measure(qubits[3], key="m4"),
            cirq.measure(qubits[4], key="m5"),
            cirq.measure(qubits[5], key="m6"),
            cirq.measure(qubits[6], key="m7"),
            cirq.measure(qubits[7], key="m8"),
            cirq.measure(qubits[8], key="m9"),
            cirq.measure(qubits[9], key="m10"),
            cirq.measure(qubits[10], key="m11"),
            cirq.measure(qubits[11], key="m12"),
            cirq.measure(qubits[12], key="m13"),
            cirq.measure(qubits[13], key="m14"),
            cirq.measure(qubits[14], key="m15"),
        ],
        # no need to measure index 15, that is our magic state
    )

    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 = sympy.symbols(
        "m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15"
    )

    # all those parities must be 0 for it to be even ( output of xor is 0 for even )
    # so stop when true == ~0 & ~0 & ~0 & ~0 == ~(0 | 0 | 0 | 0)
    even_parity = sympy.Not(
        sympy.Xor(m4, m5, m6, m7, m8, m9, m10, m11)
        | sympy.Xor(m1, m2, m3, m4, m5, m6, m7, m15)
        | sympy.Xor(m2, m3, m4, m5, m10, m11, m12, m13)
        | sympy.Xor(m1, m2, m5, m6, m9, m10, m13, m14)
    )
    # if special parity of all qubits is odd, do Z
    special_parity = sympy.Xor(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15)

    # repeating until matched parities are all even
    magic_state_cir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(),
            use_repetition_ids=False,
            repeat_until=cirq.SympyCondition(even_parity),
        ),
        # magic state "correction"
        # is odd, so do the Z
        cirq.Z(qubits[15]).with_classical_controls(special_parity),
    )

    return magic_state_cir
