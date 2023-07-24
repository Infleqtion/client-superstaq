# pylint: disable=missing-function-docstring,missing-class-docstring
import itertools
import textwrap

import cirq
import numpy as np
import pytest
import sympy

import cirq_superstaq as css


def test_zz_swap_gate() -> None:
    theta = 0.123
    gate = css.ZZSwapGate(theta)

    assert str(gate) == "ZZSwapGate(0.123)"
    assert repr(gate) == "css.ZZSwapGate(0.123)"
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")

    expected = np.array(
        [[1, 0, 0, 0], [0, 0, np.exp(1j * theta), 0], [0, np.exp(1j * theta), 0, 0], [0, 0, 0, 1]]
    )
    assert np.allclose(cirq.unitary(gate), expected)

    qubits = cirq.LineQubit.range(3)
    operation = gate(qubits[0], qubits[2])
    assert cirq.decompose_once(operation) == [
        cirq.CX(qubits[0], qubits[2]),
        cirq.CX(qubits[2], qubits[0]),
        cirq.Z(qubits[2]) ** (theta / np.pi),
        cirq.CX(qubits[0], qubits[2]),
    ]

    cirq.testing.assert_has_consistent_apply_unitary(gate)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate, ignoring_global_phase=False)
    cirq.testing.assert_consistent_resolve_parameters(gate)
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(gate)

    assert cirq.approx_eq(css.ZZSwapGate(1.23), css.ZZSwapGate(1.23 + 2 * np.pi))
    assert cirq.approx_eq(css.ZZSwapGate(np.pi + 1e-10), css.ZZSwapGate(np.pi - 1e-10))

    assert gate**1 == gate
    assert gate**-1 == css.ZZSwapGate(-0.123)

    for exponent in range(-4, 5):
        assert np.allclose(
            cirq.unitary(gate**exponent), np.linalg.matrix_power(expected, exponent)
        )
        if exponent % 2:
            assert isinstance(gate**exponent, css.ZZSwapGate)
        else:
            assert isinstance(gate**exponent, cirq.ZZPowGate)

    assert gate.__pow__(sympy.var("exponent")) is NotImplemented

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = gate**1.23


def test_zz_swap_circuit() -> None:
    qubits = cirq.LineQubit.range(3)
    operation = css.ZZSwapGate(0.456 * np.pi)(qubits[0], qubits[2])
    circuit = cirq.Circuit(operation)

    expected_diagram = textwrap.dedent(
        """
        0: ───ZZSwap(0.456π)───
              │
        2: ───ZZSwap(0.456π)───
        """
    )

    expected_qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1), q(2)]
        qreg q[3];


        zzswap(pi*0.456) q[0],q[2];
        """
    )

    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.to_qasm(header="", qubit_order=qubits) == expected_qasm

    circuit = cirq.Circuit(css.ZZSwapGate(0.0)(qubits[0], qubits[1]))
    assert circuit.to_qasm() == cirq.Circuit(cirq.SWAP(qubits[0], qubits[1])).to_qasm()


def test_zz_swap_parameterized() -> None:
    gate = css.ZZSwapGate(sympy.var("θ"))
    cirq.testing.assert_consistent_resolve_parameters(gate)

    assert gate == css.ZZSwapGate(sympy.var("θ"))
    assert cirq.approx_eq(gate, css.ZZSwapGate(sympy.var("θ")))
    assert cirq.equal_up_to_global_phase(gate, css.ZZSwapGate(sympy.var("θ")))

    with pytest.raises(TypeError, match="cirq.unitary failed. Value doesn't have"):
        _ = cirq.unitary(gate)

    with pytest.raises(TypeError, match="No Pauli expansion"):
        _ = cirq.pauli_expansion(gate)


def test_stripped_cz_gate() -> None:
    rz_rads = 0.123
    gate = css.StrippedCZGate(rz_rads)
    assert cirq.has_unitary(gate)

    assert str(gate) == "StrippedCZGate(0.123)"
    assert repr(gate) == "css.StrippedCZGate(0.123)"
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")
    expected = np.diag(
        ([1.0, np.exp(1j * rz_rads), np.exp(1j * rz_rads), np.exp(1j * (2 * rz_rads - np.pi))])
    )
    assert np.allclose(cirq.unitary(gate), expected)

    qubits = cirq.LineQubit.range(3)
    operation = gate(qubits[0], qubits[2])
    assert cirq.decompose_once(operation) == [
        cirq.rz(rz_rads).on(qubits[0]),
        cirq.rz(rz_rads).on(qubits[2]),
        cirq.CZ(qubits[0], qubits[2]),
    ]
    cirq.testing.assert_has_consistent_apply_unitary(gate)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate, ignoring_global_phase=True)
    cirq.testing.assert_consistent_resolve_parameters(gate)
    cirq.testing.assert_json_roundtrip_works(
        gate, resolvers=[*css.SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS]
    )

    assert cirq.approx_eq(css.StrippedCZGate(1.23), css.StrippedCZGate(1.23 + 2 * np.pi))
    assert cirq.approx_eq(css.StrippedCZGate(np.pi + 1e-10), css.StrippedCZGate(np.pi - 1e-10))

    phases = [
        1,
        np.exp(1j * 0.0615),
        np.exp(1j * 0.0615),
        np.exp(1j * (2 * 0.0615 - 0.5 * np.pi)),
    ]
    z_exp_gate = cirq.ZPowGate(exponent=0.246)

    assert gate**0 == cirq.IdentityGate(2)
    assert gate**0.5 == cirq.DiagonalGate(phases)
    assert gate**1 == gate
    assert gate**2 == css.ParallelGates(z_exp_gate, z_exp_gate)


def test_stripped_cz_gate_circuit() -> None:
    qubits = cirq.LineQubit.range(3)
    operation = css.StrippedCZGate(0.456 * np.pi)(qubits[0], qubits[2])
    circuit = cirq.Circuit(operation)

    expected_diagram = textwrap.dedent(
        """
        0: ───@(0.456π)───
              │
        2: ───@(0.456π)───
        """
    )

    cirq.testing.assert_has_diagram(circuit, expected_diagram)


def test_stripped_cz_gate_parameterized() -> None:
    gate = css.StrippedCZGate(sympy.var("φ"))
    cirq.testing.assert_consistent_resolve_parameters(gate)

    assert gate == css.StrippedCZGate(sympy.var("φ"))
    assert cirq.approx_eq(gate, css.StrippedCZGate(sympy.var("φ")))
    assert cirq.equal_up_to_global_phase(gate, css.StrippedCZGate(sympy.var("φ")))

    with pytest.raises(TypeError, match="cirq.unitary failed. Value doesn't have"):
        _ = cirq.unitary(gate)
    with pytest.raises(TypeError, match="No Pauli expansion"):
        _ = cirq.pauli_expansion(gate)


def test_zx_matrix() -> None:
    np.testing.assert_allclose(
        cirq.unitary(css.ZX),
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]),
    )


def test_zx_str() -> None:
    assert str(css.ZX) == "ZX"
    assert str(css.ZX**0.5) == "ZX**0.5"
    assert str(css.ZXPowGate(global_shift=0.1)) == "ZX"

    iZZ = css.ZXPowGate(global_shift=0.5)
    assert str(iZZ) == "ZX"


def test_zx_repr() -> None:
    assert repr(css.ZXPowGate()) == "css.ZX"
    assert repr(css.ZXPowGate(exponent=0.5)) == "(css.ZX**0.5)"
    assert (
        repr(css.ZXPowGate(exponent=0.5, global_shift=0.123))
        == "css.ZXPowGate(exponent=0.5, global_shift=0.123)"
    )

    cirq.testing.assert_equivalent_repr(css.ZXPowGate(), setup_code="import cirq_superstaq as css")


def test_zx_circuit() -> None:
    a, b = cirq.LineQubit.range(2)

    op = css.CR(a, b)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(op),
        textwrap.dedent(
            """
            0: ───Z───
                  │
            1: ───X───
            """
        ),
    )

    assert cirq.Circuit(op, op**0.25).to_qasm(header="") == textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1)]
        qreg q[2];


        rzx(pi*1.0) q[0],q[1];
        rzx(pi*0.25) q[0],q[1];
        """
    )


def test_acecr_init() -> None:
    css.AceCR("+-")
    css.AceCR("-+", sandwich_rx_rads=np.pi / 3)
    with pytest.raises(ValueError, match="Polarity must be"):
        css.AceCR("++")

    css.AceCR(rads=np.pi / 3)
    css.AceCR(rads=np.pi / 4)
    css.AceCR(rads=np.pi / 3, sandwich_rx_rads=np.pi)


def test_acecr_circuit_diagram_info() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(css.AceCRMinusPlus(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR-+(Z side)───
              │
        1: ───AceCR-+(X side)───"""
    )

    circuit = cirq.Circuit(css.AceCRPlusMinus(*reversed(qubits)))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR+-(X side)───
              │
        1: ───AceCR+-(Z side)───"""
    )

    circuit = cirq.Circuit(css.AceCR("+-", sandwich_rx_rads=np.pi)(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR+-(Z side)──────────
              │
        1: ───AceCR+-(X side)|Rx(π)|───"""
    )

    circuit = cirq.Circuit(css.AceCR(rads=np.pi)(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR(π)(Z side)───
              │
        1: ───AceCR(π)(X side)───"""
    )

    circuit = cirq.Circuit(css.AceCR(rads=np.pi / 3)(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR(0.333π)(Z side)───
              │
        1: ───AceCR(0.333π)(X side)───"""
    )

    circuit = cirq.Circuit(css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi)(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR(π)(Z side)──────────
              │
        1: ───AceCR(π)(X side)|Rx(π)|───"""
    )


def test_acecr_qasm() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        css.AceCR("+-").on(*qubits),
        css.AceCR("-+").on(*reversed(qubits)),
        css.AceCR("+-", sandwich_rx_rads=np.pi / 2).on(*qubits),
        css.AceCR("-+", sandwich_rx_rads=np.pi / 2).on(*qubits),
        css.AceCR(rads=np.pi / 5).on(*qubits),
        css.AceCR(rads=np.pi / 5, sandwich_rx_rads=np.pi / 2).on(*qubits),
    )

    assert circuit.to_qasm(header="") == textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1)]
        qreg q[2];


        acecr(pi*0.5) q[0],q[1];
        acecr(pi*-0.5) q[1],q[0];
        acecr_rx(pi*0.5,pi*0.5) q[0],q[1];
        acecr_rx(pi*-0.5,pi*0.5) q[0],q[1];
        acecr(pi*0.2) q[0],q[1];
        acecr_rx(pi*0.2,pi*0.5) q[0],q[1];
        """
    )


def test_acecr_eq() -> None:
    assert css.AceCRPlusMinus == css.AceCR("+-")
    assert css.AceCRPlusMinus != css.AceCR("-+")

    assert css.AceCR("+-", sandwich_rx_rads=np.pi) == css.AceCR("+-", sandwich_rx_rads=np.pi)
    assert css.AceCR("-+", sandwich_rx_rads=np.pi) != css.AceCR("+-", sandwich_rx_rads=np.pi)
    assert css.AceCR("+-", sandwich_rx_rads=np.pi) != css.AceCR("+-", sandwich_rx_rads=3 * np.pi)
    assert css.AceCR("+-", sandwich_rx_rads=np.pi) == css.AceCR("+-", sandwich_rx_rads=5 * np.pi)

    assert css.AceCR(rads=np.pi) == css.AceCR(rads=np.pi)
    assert not css.AceCR(rads=-np.pi) == css.AceCR(rads=np.pi)
    assert not css.AceCR(rads=np.pi) == css.AceCR(rads=3 * np.pi)
    assert css.AceCR(rads=np.pi) == css.AceCR(rads=5 * np.pi)
    assert css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi) == css.AceCR(
        rads=np.pi, sandwich_rx_rads=np.pi
    )

    assert not cirq.approx_eq(css.AceCR("+-"), css.AceCR("-+"))
    assert cirq.approx_eq(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=np.pi)
    )
    assert not cirq.approx_eq(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("-+", sandwich_rx_rads=np.pi)
    )
    assert not cirq.approx_eq(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=3 * np.pi)
    )
    assert cirq.approx_eq(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=5 * np.pi)
    )

    assert cirq.approx_eq(css.AceCR(rads=np.pi), css.AceCR(rads=np.pi))
    assert not cirq.approx_eq(css.AceCR(rads=np.pi), css.AceCR(rads=2 * np.pi))
    assert not cirq.approx_eq(
        css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi),
        css.AceCR(rads=2 * np.pi, sandwich_rx_rads=3 * np.pi),
    )
    assert cirq.approx_eq(
        css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi),
        css.AceCR(rads=np.pi, sandwich_rx_rads=5 * np.pi),
    )

    assert cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=np.pi)
    )
    assert not cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("-+", sandwich_rx_rads=np.pi)
    )
    assert cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=3 * np.pi)
    )
    assert cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=np.pi), css.AceCR("+-", sandwich_rx_rads=5 * np.pi)
    )
    assert cirq.equal_up_to_global_phase(css.AceCR(rads=np.pi), css.AceCR(rads=np.pi))
    assert cirq.equal_up_to_global_phase(css.AceCR(rads=-np.pi), css.AceCR(rads=np.pi))
    assert cirq.equal_up_to_global_phase(
        css.AceCR(sandwich_rx_rads=np.pi), css.AceCR(sandwich_rx_rads=np.pi)
    )
    assert not cirq.equal_up_to_global_phase(
        css.AceCR(sandwich_rx_rads=np.pi / 2), css.AceCR(sandwich_rx_rads=np.pi)
    )
    assert cirq.equal_up_to_global_phase(
        css.AceCR(sandwich_rx_rads=np.pi), css.AceCR(sandwich_rx_rads=3 * np.pi)
    )

    assert not cirq.equal_up_to_global_phase(css.AceCR("+-"), cirq.CX)
    assert not cirq.equal_up_to_global_phase(css.AceCR(rads=np.pi), cirq.CX)


def test_acecr_parameterized() -> None:
    x = sympy.var("x")
    y = sympy.var("y")

    assert cirq.is_parameterized(css.AceCR("+-", sandwich_rx_rads=x))
    assert cirq.parameter_names(css.AceCR("+-", sandwich_rx_rads=x)) == {"x"}

    assert css.AceCR("+-", sandwich_rx_rads=x) == css.AceCR("+-", sandwich_rx_rads=x)
    assert css.AceCR("+-", sandwich_rx_rads=x) != css.AceCR("-+", sandwich_rx_rads=x)
    assert css.AceCR("+-", sandwich_rx_rads=x) != css.AceCR("-+")
    assert css.AceCR(rads=x) == css.AceCR(rads=x)
    assert css.AceCR(rads=x) != css.AceCR(rads=-x)
    assert css.AceCR(rads=x) != css.AceCR()
    assert css.AceCR(rads=x, sandwich_rx_rads=y) == css.AceCR(rads=x, sandwich_rx_rads=y)
    assert css.AceCR(rads=x, sandwich_rx_rads=y) != css.AceCR(rads=-x, sandwich_rx_rads=y)

    assert cirq.approx_eq(css.AceCR("+-", sandwich_rx_rads=x), css.AceCR("+-", sandwich_rx_rads=x))
    assert not cirq.approx_eq(
        css.AceCR("+-", sandwich_rx_rads=x), css.AceCR("-+", sandwich_rx_rads=x)
    )

    assert cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=x), css.AceCR("+-", sandwich_rx_rads=x)
    )
    assert not cirq.equal_up_to_global_phase(
        css.AceCR("+-", sandwich_rx_rads=x), css.AceCR("-+", sandwich_rx_rads=x)
    )
    assert not cirq.equal_up_to_global_phase(css.AceCR("+-", sandwich_rx_rads=x), css.AceCR("-+"))

    cirq.testing.assert_consistent_resolve_parameters(css.AceCR("+-", sandwich_rx_rads=x))
    cirq.testing.assert_consistent_resolve_parameters(css.AceCR("-+", sandwich_rx_rads=x))


def test_acecr_repr_and_str() -> None:
    assert repr(css.AceCRPlusMinus) == "css.AceCR()"
    assert repr(css.AceCRMinusPlus) == "css.AceCR(rads=-1.5707963267948966)"
    assert repr(css.AceCR(rads=np.pi)) == "css.AceCR(rads=3.141592653589793)"
    assert (
        repr(css.AceCR(sandwich_rx_rads=np.pi)) == "css.AceCR(sandwich_rx_rads=3.141592653589793)"
    )
    assert (
        repr(css.AceCR(rads=np.pi / 3, sandwich_rx_rads=np.pi / 2))
        == "css.AceCR(rads=1.0471975511965976, sandwich_rx_rads=1.5707963267948966)"
    )
    assert (
        repr(css.AceCR("+-", sandwich_rx_rads=np.pi))
        == "css.AceCR(sandwich_rx_rads=3.141592653589793)"
    )

    cirq.testing.assert_equivalent_repr(
        css.AceCRPlusMinus, setup_code="import cirq_superstaq as css"
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCRMinusPlus, setup_code="import cirq_superstaq as css"
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCR(np.pi),
        setup_code="import cirq; import cirq_superstaq as css",
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCR(sandwich_rx_rads=np.pi),
        setup_code="import cirq; import cirq_superstaq as css",
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCR(np.pi, np.pi / 2), setup_code="import cirq_superstaq as css"
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCR("+-", sandwich_rx_rads=np.pi),
        setup_code="import cirq; import cirq_superstaq as css",
    )

    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCRPlusMinus, ignoring_global_phase=False
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCRMinusPlus, ignoring_global_phase=False
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCR(np.pi / 5), ignoring_global_phase=False
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCR(sandwich_rx_rads=np.pi / 2), ignoring_global_phase=False
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCR(np.pi, np.pi / 2), ignoring_global_phase=False
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        css.AceCR("-+", np.pi / 2), ignoring_global_phase=False
    )

    assert str(css.AceCR()) == "AceCR"
    assert str(css.AceCR("+-", sandwich_rx_rads=np.pi)) == "AceCR|Rx(π)|"
    assert str(css.AceCR(rads=np.pi)) == "AceCR(3.141592653589793)"
    assert str(css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi)) == "AceCR(3.141592653589793)|Rx(π)|"


def test_acecr_decompose() -> None:
    a = cirq.LineQubit(0)
    b = cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.decompose_once(css.AceCRMinusPlus(a, b)))
    assert len(circuit) == 3 and len(list(circuit.all_operations())) == 3

    circuit = cirq.Circuit(cirq.decompose_once(css.AceCR("+-", sandwich_rx_rads=-np.pi / 2)(a, b)))
    assert len(circuit) == 3 and len(list(circuit.all_operations())) == 4


def test_barrier() -> None:
    n = 3
    qubits = cirq.LineQubit.range(n)
    gate = css.Barrier(n)

    assert gate == css.Barrier(n)
    assert gate != cirq.IdentityGate(n)

    assert str(gate) == "Barrier(3)"
    assert repr(gate) == "css.Barrier(3)"

    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")

    operation = gate.on(*qubits)
    assert cirq.decompose(operation) == [operation]

    # confirm Barrier is an InterchangeableQubitsGate
    for permuted_qubits in itertools.permutations(qubits):
        assert operation == gate.on(*permuted_qubits)

    circuit = cirq.Circuit(operation)
    expected_qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1), q(2)]
        qreg q[3];


        barrier q[0],q[1],q[2];
        """
    )
    assert circuit.to_qasm(header="") == expected_qasm

    cirq.testing.assert_has_diagram(
        circuit,
        textwrap.dedent(
            """
            0: ───│───
                  │
            1: ───│───
                  │
            2: ───│───
            """
        ),
        use_unicode_characters=True,
    )

    cirq.testing.assert_has_diagram(
        circuit,
        textwrap.dedent(
            """
            0: ---|---
                  |
            1: ---|---
                  |
            2: ---|---
            """
        ),
        use_unicode_characters=False,
    )

    # make sure optimizations don't drop Barriers:
    circuit = cirq.drop_negligible_operations(circuit)
    assert circuit == cirq.Circuit(operation)
    assert cirq.trace_distance_bound(gate) == 1.0

    barrier = css.barrier(*qubits)
    assert barrier == css.Barrier(n).on(*qubits)


def test_barrier_on_qids() -> None:
    qudits = [
        cirq.LineQubit(0),
        cirq.LineQid(1, 3),
        cirq.GridQid(2, 3, dimension=4),
        cirq.NamedQid("foo", dimension=5),
    ]

    gate = css.Barrier(qid_shape=(2, 3, 4, 5))

    assert str(gate) == "Barrier(qid_shape=(2, 3, 4, 5))"
    assert repr(gate) == "css.Barrier(qid_shape=(2, 3, 4, 5))"

    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")
    assert gate != css.Barrier(4)

    operation = gate.on(*qudits)
    assert cirq.decompose(operation) == [operation]

    # make sure optimizations don't drop Barriers:
    circuit = cirq.drop_negligible_operations(cirq.Circuit(operation))
    assert circuit == cirq.Circuit(operation)
    assert cirq.trace_distance_bound(operation) == 1.0

    # check css.barrier() function and confirm Barrier is an InterchangeableQubitsGate
    # (only works if all qudits have the same dimension)
    qudits = [q.with_dimension(3) for q in qudits]
    operation = css.barrier(*qudits)
    for permuted_qubits in itertools.permutations(qudits):
        qid_shape = tuple(q.dimension for q in permuted_qubits)
        assert operation == css.Barrier(qid_shape=qid_shape).on(*permuted_qubits)
        assert operation == css.barrier(*permuted_qubits)


def test_parallel_gates() -> None:
    gate = css.ParallelGates(cirq.CZ, cirq.CZ**0.5, cirq.CZ**-0.5)
    qubits = cirq.LineQubit.range(6)
    operation = gate(*qubits)
    circuit = cirq.Circuit(operation)

    expected_diagram = textwrap.dedent(
        """
        0: ───@₁────────
              │
        1: ───@₁────────
              │
        2: ───@₂────────
              │
        3: ───@₂^0.5────
              │
        4: ───@₃────────
              │
        5: ───@₃^-0.5───
        """
    )
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq, cirq_superstaq as css")
    assert repr(gate) == "css.ParallelGates(cirq.CZ, (cirq.CZ**0.5), (cirq.CZ**-0.5))"
    assert str(gate) == "ParallelGates(CZ, CZ**0.5, CZ**-0.5)"

    assert cirq.decompose(operation) == [
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[2], qubits[3]) ** 0.5,
        cirq.CZ(qubits[4], qubits[5]) ** -0.5,
    ]
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate, ignoring_global_phase=False)

    assert gate**0.5 == css.ParallelGates(cirq.CZ**0.5, cirq.CZ**0.25, cirq.CZ**-0.25)

    assert css.ParallelGates(cirq.X, cirq.Y, cirq.Z) == css.ParallelGates(
        cirq.X, css.ParallelGates(cirq.Y, cirq.Z)
    )

    assert css.ParallelGates(cirq.X, cirq.Y, cirq.Y) == css.ParallelGates(
        cirq.X, cirq.ParallelGate(cirq.Y, num_copies=2)
    )

    with pytest.raises(ValueError, match="ParallelGates cannot contain measurements"):
        _ = css.ParallelGates(cirq.X, cirq.MeasurementGate(1, key="1"))

    with pytest.raises(ValueError, match="is not a cirq Gate"):
        _ = css.ParallelGates(cirq.X(qubits[1]))


def test_parallel_gates_operation() -> None:
    q0, _, q2, q3 = cirq.LineQubit.range(4)
    op = css.parallel_gates_operation(cirq.CX(q2, q0), cirq.Y(q3))
    assert op == css.ParallelGates(cirq.CX, cirq.Y).on(q2, q0, q3)

    with pytest.raises(ValueError, match="no .gate attribute"):
        _ = css.parallel_gates_operation(cirq.X(q0).with_classical_controls("1"))

    with pytest.raises(ValueError, match="tagged operations not permitted"):
        _ = css.parallel_gates_operation(cirq.X(q0).with_tags("foo"))

    with pytest.raises(ValueError):  # Overlapping qubits should be caught by cirq
        _ = css.parallel_gates_operation(cirq.CX(q2, q0), cirq.Y(q2))


def test_parallel_gates_on_qids() -> None:
    gate = css.ParallelGates(cirq.X, cirq.MatrixGate(np.eye(9), qid_shape=(3, 3)))
    qudits = [cirq.LineQubit(2), cirq.LineQid(1, 3), cirq.GridQid(2, 3, dimension=3)]
    operation = gate(*qudits)

    cirq.testing.assert_equivalent_repr(
        gate, setup_code="import cirq, cirq_superstaq as css, numpy as np"
    )

    assert cirq.decompose(operation) == [
        cirq.X(qudits[0]),
        cirq.MatrixGate(np.eye(9), qid_shape=(3, 3)).on(qudits[1], qudits[2]),
    ]
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate, ignoring_global_phase=False)

    assert operation == css.parallel_gates_operation(*cirq.decompose(operation))


def test_parallel_gates_circuit_diagram_fallback() -> None:
    gate = cirq.circuits.qasm_output.QasmUGate(0.1, 0.2, 0.3)
    assert not hasattr(gate, "_circuit_diagram_info_")

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.ParallelGates(gate).on(cirq.LineQubit(1))),
        f"1: ───ParallelGates({gate})───",
    )


def test_parallel_gates_eq() -> None:
    gate = css.ParallelGates(cirq.X, cirq.ry(2.1))

    assert gate == css.ParallelGates(cirq.X, cirq.ry(2.1))
    assert gate != css.ParallelGates(cirq.X, cirq.ry(2.1 + 2 * np.pi))
    assert gate == css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi))
    assert gate != css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi + 1e-10))
    assert gate != css.ParallelGates(cirq.rx(np.pi), cirq.ry(2.1))

    assert cirq.approx_eq(gate, css.ParallelGates(cirq.X, cirq.ry(2.1)))
    assert not cirq.approx_eq(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 2 * np.pi)))
    assert cirq.approx_eq(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi)))
    assert cirq.approx_eq(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi + 1e-10)))
    assert not cirq.approx_eq(gate, css.ParallelGates(cirq.rx(np.pi), cirq.ry(2.1)))

    assert cirq.equal_up_to_global_phase(gate, css.ParallelGates(cirq.X, cirq.ry(2.1)))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 2 * np.pi)))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi)))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelGates(cirq.X, cirq.ry(2.1 + 4 * np.pi)))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelGates(cirq.rx(np.pi), cirq.ry(2.1)))

    assert not cirq.equal_up_to_global_phase(css.ParallelGates(), cirq.X)
    assert not cirq.equal_up_to_global_phase(css.ParallelGates(), css.ParallelGates(cirq.X))
    assert not cirq.equal_up_to_global_phase(css.ParallelGates(cirq.CX), css.ParallelGates(cirq.X))


def test_parallel_gates_parameterized() -> None:
    x = sympy.var("x")
    y = sympy.var("y")
    gate = css.ParallelGates(cirq.X**x, cirq.ry(y))

    assert cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == {"x", "y"}

    assert gate == css.ParallelGates(cirq.X**x, cirq.ry(y))
    assert gate != css.ParallelGates(cirq.X**y, cirq.ry(x))
    assert cirq.approx_eq(gate, css.ParallelGates(cirq.X**x, cirq.ry(y)))

    cirq.testing.assert_consistent_resolve_parameters(css.ParallelGates(cirq.X**x, cirq.ry(y)))
    cirq.testing.assert_consistent_resolve_parameters(css.ParallelGates(cirq.X**x, cirq.ry(x)))


def test_parallel_gates_equivalence_groups() -> None:
    qubits = cirq.LineQubit.range(4)
    gate = css.ParallelGates(cirq.X, css.ZX, cirq.Y)
    operation = gate(*qubits[:4])
    assert [gate.qubit_index_to_equivalence_group_key(i) for i in range(4)] == [0, 1, 2, 3]
    for permuted_qubits in itertools.permutations(operation.qubits):
        if permuted_qubits == operation.qubits:
            assert operation == gate(*permuted_qubits)
        else:
            assert operation != gate(*permuted_qubits)

    gate = css.ParallelGates(cirq.X, cirq.X, css.ZZSwapGate(1.23))
    operation = gate(*qubits)
    assert [gate.qubit_index_to_equivalence_group_key(i) for i in range(4)] == [0, 0, 2, 2]

    equivalent_targets = [
        (qubits[0], qubits[1], qubits[2], qubits[3]),
        (qubits[1], qubits[0], qubits[2], qubits[3]),
        (qubits[0], qubits[1], qubits[3], qubits[2]),
        (qubits[1], qubits[0], qubits[3], qubits[2]),
    ]
    for permuted_qubits in itertools.permutations(operation.qubits):
        if permuted_qubits in equivalent_targets:
            assert operation == gate(*permuted_qubits)
            assert cirq.approx_eq(operation, gate(*permuted_qubits))
            assert cirq.equal_up_to_global_phase(operation, gate(*permuted_qubits))
        else:
            assert operation != gate(*permuted_qubits)
            assert not cirq.approx_eq(operation, gate(*permuted_qubits))
            assert not cirq.equal_up_to_global_phase(operation, gate(*permuted_qubits))

    with pytest.raises(ValueError, match="index out of range"):
        _ = gate.qubit_index_to_equivalence_group_key(4)

    with pytest.raises(ValueError, match="index out of range"):
        _ = gate.qubit_index_to_equivalence_group_key(-1)


def test_parallel_gates_equivalence_groups_nonadjacent() -> None:
    qubits = cirq.LineQubit.range(4)
    gate = css.ParallelGates(cirq.X, css.ZZSwapGate(1.23), cirq.X)
    assert [gate.qubit_index_to_equivalence_group_key(i) for i in range(4)] == [0, 1, 1, 0]

    operation = gate(*qubits)
    equivalent_targets = [
        (qubits[0], qubits[1], qubits[2], qubits[3]),
        (qubits[0], qubits[2], qubits[1], qubits[3]),
        (qubits[3], qubits[1], qubits[2], qubits[0]),
        (qubits[3], qubits[2], qubits[1], qubits[0]),
    ]
    for permuted_qubits in itertools.permutations(operation.qubits):
        if permuted_qubits in equivalent_targets:
            assert operation == gate(*permuted_qubits)
            assert cirq.approx_eq(operation, gate(*permuted_qubits))
            assert cirq.equal_up_to_global_phase(operation, gate(*permuted_qubits))
        else:
            assert operation != gate(*permuted_qubits)
            assert not cirq.approx_eq(operation, gate(*permuted_qubits))
            assert not cirq.equal_up_to_global_phase(operation, gate(*permuted_qubits))


def test_rgate() -> None:
    qubit = cirq.LineQubit(0)

    rot_gate = css.RGate(4.56 * np.pi, 1.23 * np.pi)
    cirq.testing.assert_equivalent_repr(rot_gate, setup_code="import cirq_superstaq as css")
    assert str(rot_gate) == f"RGate({rot_gate.exponent}π, {rot_gate.phase_exponent}π)"
    assert rot_gate**-1 == css.RGate(-rot_gate.theta, rot_gate.phi)

    circuit = cirq.Circuit(rot_gate.on(qubit))

    # build RGate decomposition manually
    decomposed_circuit = cirq.Circuit(
        cirq.rz(-rot_gate.phi).on(qubit),
        cirq.rx(rot_gate.theta).on(qubit),
        cirq.rz(+rot_gate.phi).on(qubit),
    )

    assert np.allclose(cirq.unitary(circuit), cirq.unitary(decomposed_circuit))

    expected_qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0)]
        qreg q[1];


        r(pi*4.56,pi*-0.77) q[0];
        """
    )
    assert circuit.to_qasm(header="") == expected_qasm

    circuit = cirq.Circuit(css.RGate(np.pi, 0.5 * np.pi).on(qubit))
    cirq.testing.assert_has_diagram(circuit, "0: ───RGate(π, 0.5π)───")


def test_rgate_eq() -> None:
    gate = css.RGate(5 * np.pi / 8, 1.23)

    assert gate == css.RGate(gate.theta, gate.phi)
    assert gate != css.RGate(gate.theta + 2 * np.pi, gate.phi)
    assert gate == css.RGate(gate.theta + 4 * np.pi, gate.phi)
    assert gate != cirq.PhasedXPowGate(exponent=gate.exponent, phase_exponent=gate.phase_exponent)
    assert gate == cirq.PhasedXPowGate(
        exponent=gate.exponent, phase_exponent=gate.phase_exponent, global_shift=-0.5
    )

    assert cirq.equal_up_to_global_phase(gate, css.RGate(gate.theta + 2 * np.pi, gate.phi))
    assert cirq.equal_up_to_global_phase(gate, css.RGate(-gate.theta, gate.phi + np.pi))
    assert cirq.equal_up_to_global_phase(
        gate, cirq.PhasedXPowGate(exponent=gate.exponent, phase_exponent=gate.phase_exponent)
    )
    assert cirq.equal_up_to_global_phase(
        gate, cirq.PhasedXPowGate(exponent=gate.exponent + 2, phase_exponent=gate.phase_exponent)
    )

    assert not cirq.equal_up_to_global_phase(gate, css.RGate(gate.theta, gate.phi + np.pi))
    assert not cirq.equal_up_to_global_phase(gate, css.RGate(-gate.theta, gate.phi))
    assert not cirq.equal_up_to_global_phase(gate, cirq.CX)


def test_rgate_parameterized() -> None:
    x = sympy.var("x")
    y = sympy.var("y")
    gate = css.RGate(x, y)

    assert cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == {"x", "y"}

    assert gate == css.RGate(x, y)
    assert gate != css.RGate(x, x)
    assert gate != css.RGate(y, x)

    assert cirq.approx_eq(gate, css.RGate(x, y))
    assert cirq.equal_up_to_global_phase(gate, css.RGate(x, y))
    assert not cirq.equal_up_to_global_phase(gate, css.RGate(x, x))
    assert not cirq.equal_up_to_global_phase(gate, css.RGate(y, x))

    cirq.testing.assert_consistent_resolve_parameters(css.RGate(x, y))
    cirq.testing.assert_consistent_resolve_parameters(css.RGate(x, x))


def test_parallel_rgate() -> None:
    qubits = cirq.LineQubit.range(2)

    rot_gate = css.ParallelRGate(1.23 * np.pi, 4.56 * np.pi, len(qubits))
    cirq.testing.assert_equivalent_repr(
        rot_gate, setup_code="import cirq; import cirq_superstaq as css"
    )
    text = f"RGate({rot_gate.exponent}π, {rot_gate.phase_exponent}π) x {len(qubits)}"
    assert str(rot_gate) == text
    assert rot_gate**-1 == css.ParallelRGate(-rot_gate.theta, rot_gate.phi, len(qubits))

    circuit = cirq.Circuit(rot_gate.on(*qubits))

    # build ParallelRGate decomposition manually
    manual_circuit = cirq.Circuit(
        [css.RGate(rot_gate.theta, rot_gate.phi).on(qubit) for qubit in qubits]
    )

    assert np.allclose(cirq.unitary(circuit), cirq.unitary(manual_circuit))

    expected_qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1)]
        qreg q[2];


        gate_GR(pi*1.23,pi*0.56) q[0],q[1];
        """
    )
    assert circuit.to_qasm(header="") == expected_qasm

    circuit = cirq.Circuit(css.ParallelRGate(np.pi, 0.5 * np.pi, len(qubits)).on(*qubits))
    expected_diagram = textwrap.dedent(
        """
        0: ───RGate(π, 0.5π)───
              │
        1: ───#2───────────────
        """
    )

    expected_qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1)]
        qreg q[2];


        gate_GR(pi*1.0,pi*0.5) q[0],q[1];
        """
    )

    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.to_qasm(header="", qubit_order=qubits) == expected_qasm


def test_parallel_rgate_eq() -> None:
    theta, phi = 4.2, 1.7
    gate = css.ParallelRGate(theta, phi, 3)

    assert gate == css.ParallelRGate(theta, phi, 3)
    assert gate != css.ParallelRGate(theta + 2 * np.pi, phi, 3)
    assert gate == css.ParallelRGate(theta + 4 * np.pi, phi, 3)

    assert cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta, phi, 3))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta + 2 * np.pi, phi, 3))
    assert cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta + 4 * np.pi, phi, 3))
    assert cirq.equal_up_to_global_phase(gate, cirq.ParallelGate(gate.sub_gate, 3))

    assert not cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta + 1, phi, 3))
    assert not cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta, phi + 1, 3))
    assert not cirq.equal_up_to_global_phase(gate, css.ParallelRGate(theta, phi, 2))
    assert not cirq.equal_up_to_global_phase(gate, cirq.CX)


def test_ixgate() -> None:
    gate = css.ops.qubit_gates.IXGate()

    assert str(gate) == "IX"
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")
    cirq.testing.assert_consistent_resolve_parameters(gate)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gate(cirq.LineQubit(1))),
        textwrap.dedent(
            """
            1: ───iX───
            """
        ),
    )

    assert isinstance(gate**1, css.ops.qubit_gates.IXGate)
    assert isinstance(gate**5, css.ops.qubit_gates.IXGate)
    assert isinstance(gate**1.5, cirq.XPowGate)
    assert not isinstance(gate**1.5, css.ops.qubit_gates.IXGate)

    assert np.allclose(
        cirq.unitary(gate),
        np.array(
            [
                [0, 1j],
                [1j, 0],
            ]
        ),
    )


def test_itoffoli() -> None:
    qubits = cirq.LineQubit.range(3)

    assert np.allclose(
        cirq.unitary(css.ops.qubit_gates.ICCX(*qubits)),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j],
                [0, 0, 0, 0, 0, 0, 1j, 0],
            ]
        ),
    )

    assert np.allclose(
        cirq.unitary(css.AQTITOFFOLI(*qubits)),
        np.array(
            [
                [0, 1j, 0, 0, 0, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    )


def test_custom_resolver() -> None:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit += css.ZZSwapGate(1.23).on(qubits[0], qubits[1])
    circuit += css.AceCRPlusMinus(qubits[0], qubits[1])
    circuit += css.Barrier(2).on(qubits[0], qubits[1])
    circuit += css.CR(qubits[0], qubits[1])
    circuit += css.AceCRMinusPlus(qubits[0], qubits[1])
    circuit += css.AceCR("+-", sandwich_rx_rads=np.pi / 2)(qubits[0], qubits[1])
    circuit += css.AceCR(rads=np.pi / 3)(qubits[0], qubits[1])
    circuit += css.AceCR(rads=np.pi, sandwich_rx_rads=np.pi)(qubits[0], qubits[1])
    circuit += css.ParallelGates(cirq.X, css.ZX).on(qubits[0], qubits[2], qubits[3])
    circuit += cirq.ms(1.23).on(qubits[0], qubits[1])
    circuit += css.RGate(1.23, 4.56).on(qubits[0])
    circuit += css.ParallelRGate(1.23, 4.56, len(qubits)).on(*qubits)
    circuit += css.AQTITOFFOLI(qubits[0], qubits[1], qubits[2])
    circuit += css.ops.qubit_gates.ICCX(qubits[0], qubits[1], qubits[2])
    circuit += css.ops.qubit_gates.IX(qubits[0])
    circuit += css.StrippedCZGate(0.123).on(qubits[0], qubits[1])

    json_text = cirq.to_json(circuit)
    resolvers = [*css.SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS]
    assert cirq.read_json(json_text=json_text, resolvers=resolvers) == circuit
