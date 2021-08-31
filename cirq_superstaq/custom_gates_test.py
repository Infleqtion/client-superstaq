import cirq
import numpy as np

import cirq_superstaq


def test_fermionic_swap_gate() -> None:
    theta = 0.123
    gate = cirq_superstaq.FermionicSWAPGate(theta)

    assert str(gate) == "FermionicSWAPGate(0.123)"
    assert repr(gate) == "cirq_superstaq.custom_gates.FermionicSWAPGate(0.123)"
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq")

    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(gate(qubits[0], qubits[2]))

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───FermionicSWAP(0.0392π)───
      │
2: ───FermionicSWAP(0.0392π)───
""",
    )

    expected = np.array(
        [[1, 0, 0, 0], [0, 0, np.exp(1j * theta), 0], [0, np.exp(1j * theta), 0, 0], [0, 0, 0, 1]]
    )
    assert np.allclose(cirq.unitary(gate), expected)


def test_zx_matrix() -> None:
    np.testing.assert_allclose(
        cirq.unitary(cirq_superstaq.ZX),
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]),
    )


def test_zx_str() -> None:
    assert str(cirq_superstaq.ZX) == "ZX"
    assert str(cirq_superstaq.ZX ** 0.5) == "ZX**0.5"
    assert str(cirq_superstaq.ZXPowGate(global_shift=0.1)) == "ZX"

    iZZ = cirq_superstaq.ZXPowGate(global_shift=0.5)
    assert str(iZZ) == "ZX"


def test_zx_repr() -> None:
    assert repr(cirq_superstaq.ZXPowGate()) == "cirq_superstaq.ZX"
    assert repr(cirq_superstaq.ZXPowGate(exponent=0.5)) == "(cirq_superstaq.ZX**0.5)"
    assert (
        repr(cirq_superstaq.ZXPowGate(exponent=0.5, global_shift=0.123))
        == "cirq_superstaq.ZXPowGate(exponent=0.5, global_shift=0.123)"
    )

    cirq.testing.assert_equivalent_repr(
        cirq_superstaq.ZXPowGate(), setup_code="import cirq_superstaq"
    )


def test_zx_circuit() -> None:
    a, b = cirq.LineQubit.range(2)

    c = cirq.Circuit(cirq_superstaq.CR(a, b))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───Z───
      │
1: ───X───
    """,
    )


def test_acecr() -> None:
    qubits = cirq.LineQubit.range(2)
    expected = "0: ───AceCR-+(Z side)───\n      │\n1: ───AceCR-+(X side)───"
    assert str(cirq.Circuit(cirq_superstaq.AceCRMinusPlus(qubits[0], qubits[1]))) == expected
    expected = "0: ───AceCR+-(X side)───\n      │\n1: ───AceCR+-(Z side)───"
    assert str(cirq.Circuit(cirq_superstaq.AceCRPlusMinus(qubits[1], qubits[0]))) == expected
    assert cirq_superstaq.AceCRPlusMinus == cirq_superstaq.AceCR("+-")
    assert cirq_superstaq.AceCRPlusMinus != cirq_superstaq.AceCR("-+")
    assert repr(cirq_superstaq.AceCRMinusPlus) == "cirq_superstaq.AceCR('-+')"
    cirq.testing.assert_equivalent_repr(
        cirq_superstaq.AceCRMinusPlus, setup_code="import cirq_superstaq"
    )
    cirq.testing.assert_equivalent_repr(
        cirq_superstaq.AceCRPlusMinus, setup_code="import cirq_superstaq"
    )
    assert str(cirq_superstaq.AceCRMinusPlus) == "AceCR-+"
    assert hash(cirq_superstaq.AceCRMinusPlus) == hash("-+")
    assert cirq_superstaq.AceCRPlusMinus != cirq.CNOT


def test_acecr_decompose() -> None:
    a = cirq.LineQubit(0)
    b = cirq.LineQubit(1)
    assert cirq.decompose_once(cirq_superstaq.AceCRMinusPlus(a, b)) is not None


def test_barrier() -> None:
    n = 3
    gate = cirq_superstaq.Barrier(n)

    assert str(gate) == "Barrier(3)"
    assert repr(gate) == "cirq_superstaq.custom_gates.Barrier(3)"

    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq")

    operation = gate.on(*cirq.LineQubit.range(3))
    assert cirq.decompose(operation) == [operation]

    circuit = cirq.Circuit(operation)
    expected_qasm = f"""// Generated from Cirq v{cirq.__version__}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [0, 1, 2]
qreg q[3];


barrier q[0],q[1],q[2];
"""
    assert cirq.qasm(circuit) == expected_qasm

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───|───
      │
1: ───|───
      │
2: ───|───
""",
    )


def test_custom_resolver() -> None:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit += cirq_superstaq.FermionicSWAPGate(1.23).on(qubits[0], qubits[1])
    circuit += cirq_superstaq.AceCRPlusMinus(qubits[0], qubits[1])
    circuit += cirq_superstaq.Barrier(2).on(qubits[0], qubits[1])
    circuit += cirq_superstaq.CR(qubits[0], qubits[1])
    circuit += cirq_superstaq.AceCRMinusPlus(qubits[0], qubits[1])
    circuit += cirq.CX(qubits[0], qubits[1])

    json_text = cirq.to_json(circuit)
    resolvers = [cirq_superstaq.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]

    assert cirq.read_json(json_text=json_text, resolvers=resolvers) == circuit
