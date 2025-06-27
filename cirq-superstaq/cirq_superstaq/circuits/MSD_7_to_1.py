import cirq
import sympy
from cirq.circuits import InsertStrategy


def msd_7_to_1(qubits: list[cirq.LineQubit]) -> cirq.Circuit:
    """Function to perform a 7-to-1 magic state distillation protocol.

    Args:
        qubits: The list of LineQubits of length 8.
                The first qubit will be the final magic state qubit.

    Returns:
        The magic state distillation circuit.
    """
    cir = cirq.Circuit()

    for q in qubits:
        cir.append([cirq.R(q)])

    cir.append(
        [
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[4]),
        ],
        strategy=InsertStrategy.NEW_THEN_INLINE,
    )

    cir.append([cirq.CNOT(qubits[0], qubits[3])], strategy=InsertStrategy.NEW_THEN_INLINE)

    cir.append(
        [
            cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6]),
        ],
        strategy=InsertStrategy.NEW_THEN_INLINE,
    )

    cir.append(
        [
            cirq.CNOT(qubits[4], qubits[5]),
            cirq.CNOT(qubits[4], qubits[6]),
            cirq.CNOT(qubits[4], qubits[7]),
        ],
        strategy=InsertStrategy.NEW_THEN_INLINE,
    )

    cir.append(
        [
            cirq.CNOT(qubits[2], qubits[3]),
            cirq.CNOT(qubits[2], qubits[6]),
            cirq.CNOT(qubits[2], qubits[7]),
        ],
        strategy=InsertStrategy.NEW_THEN_INLINE,
    )

    cir.append(
        [
            cirq.CNOT(qubits[1], qubits[3]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[7]),
        ],
        strategy=InsertStrategy.NEW_THEN_INLINE,
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
        strategy=InsertStrategy.NEW_THEN_INLINE,
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
        strategy=InsertStrategy.NEW_THEN_INLINE,
    )

    # index 0=magic
    m0, m1, m2, m3, m4, m5, m6 = sympy.symbols("m0 m1 m2 m3 m4 m5 m6")
    # all those parities must be 0 for it to be even ( so output of xor is 0 for even )
    # so stop when true == ~0 & ~0 & ~0 == ~(0 | 0 | 0)
    # use demorgans
    # https://arxiv.org/pdf/0803.0272 page 12 figure 19
    evenParity = sympy.Not(
        sympy.Xor(m0, m1, m2, m3) | sympy.Xor(m0, m1, m4, m5) | sympy.Xor(m0, m2, m4, m6)
    )
    # if special parity is EVEN, do Z
    specialParity = sympy.Xor(m4, m5, m6)

    # repeating until matched parities
    magicStateCir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(),
            use_repetition_ids=False,
            repeat_until=cirq.SympyCondition(evenParity),
        )
    )

    # magic state "correction"
    # is even, so do the Z
    magicStateCir.append([cirq.Z(qubits[0]).with_classical_controls(sympy.Not(specialParity))])

    return magicStateCir
