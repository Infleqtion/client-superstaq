# pylint: disable=missing-function-docstring,missing-class-docstring
import functools
import textwrap
from typing import List, Optional, Tuple, Type, cast
from unittest import mock

import cirq
import numpy as np
import pytest
import scipy.linalg

import cirq_superstaq as css


@pytest.mark.parametrize("dimension", [2, 3, 4, 5, 6])
def test_qudit_swap_gate(dimension: int) -> None:
    gate = css.QuditSwapGate(dimension=dimension)
    assert gate == css.QuditSwapGate(dimension)
    assert gate != css.QuditSwapGate(dimension + 1)
    assert gate != cirq.SWAP

    assert cirq.approx_eq(gate, css.QuditSwapGate(dimension))
    assert not cirq.approx_eq(gate, css.QuditSwapGate(dimension + 1))
    assert not cirq.approx_eq(gate, cirq.SWAP)

    assert cirq.equal_up_to_global_phase(gate, css.QuditSwapGate(dimension))
    assert not cirq.equal_up_to_global_phase(gate, css.QuditSwapGate(dimension + 1))
    if dimension == 2:
        assert cirq.equal_up_to_global_phase(gate, cirq.SWAP)
    else:
        assert not cirq.equal_up_to_global_phase(gate, cirq.SWAP)

    for exponent in (-3, -1, 1, 5):
        assert gate**exponent == gate

    for exponent in (-2, 0, 2, 10):
        assert gate**exponent == cirq.IdentityGate(qid_shape=(dimension, dimension))

    assert cirq.pow(gate, 1.23, default=None) is None

    assert str(gate) == f"SWAP{dimension}"

    if hasattr(css, f"SWAP{dimension}"):
        assert repr(gate) == f"css.SWAP{dimension}"
    else:
        assert repr(gate) == f"css.QuditSwapGate(dimension={dimension})"

    assert cirq.has_unitary(gate)
    assert not cirq.is_parameterized(gate)
    assert cirq.trace_distance_bound(gate) == 1.0

    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")
    cirq.testing.assert_json_roundtrip_works(gate, resolvers=css.SUPERSTAQ_RESOLVERS)

    q0, q1 = cirq.LineQid.range(2, dimension=dimension)
    swap_op = gate.on(q0, q1)

    # Check the unitary by commuting through random single-qudit gates:
    one_qb_gate0 = cirq.MatrixGate(cirq.testing.random_unitary(dimension), qid_shape=(dimension,))
    one_qb_gate1 = cirq.MatrixGate(cirq.testing.random_unitary(dimension), qid_shape=(dimension,))
    assert np.allclose(
        cirq.Circuit(one_qb_gate0(q0), one_qb_gate1(q1), swap_op).unitary(),
        cirq.Circuit(swap_op, one_qb_gate0(q1), one_qb_gate1(q0)).unitary(),
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(swap_op),
        textwrap.dedent(
            f"""
            0 (d={dimension}): ───×───
                        │
            1 (d={dimension}): ───×───
            """
        ),
    )


def test_qudit_swap_op() -> None:
    qubit0 = cirq.LineQubit(0)
    qubit1 = cirq.GridQubit(1, 2)
    qudit0 = cirq.LineQid(0, dimension=7)
    qudit1 = cirq.GridQid(1, 2, dimension=7)

    assert css.qudit_swap_op(qubit0, qubit1) == cirq.SWAP(qubit0, qubit1)
    assert css.qudit_swap_op(qudit0, qudit1) == css.QuditSwapGate(7).on(qudit0, qudit1)
    with pytest.raises(ValueError, match="do not have the same dimension"):
        _ = css.qudit_swap_op(qubit0, qudit1)


def test_bswap_pow_gate() -> None:
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        css.BSwapPowGate,
        setup_code="import cirq_superstaq as css, sympy",
        ignore_decompose_to_default_gateset=True,
    )

    shifted_bswap = css.BSwapPowGate(global_shift=0.3)

    assert css.BSWAP == css.BSwapPowGate()
    assert css.BSWAP_INV == css.BSWAP**-1 == css.BSWAP**3

    assert css.BSWAP != shifted_bswap
    assert not cirq.approx_eq(css.BSWAP, shifted_bswap)
    assert cirq.equal_up_to_global_phase(css.BSWAP, shifted_bswap)
    assert cirq.equal_up_to_global_phase(css.BSWAP**1.23, shifted_bswap**5.23)
    assert not cirq.equal_up_to_global_phase(css.BSWAP, css.CZ3)

    assert repr(css.BSWAP) == "css.BSWAP"
    assert repr(css.BSWAP**-1) == "css.BSWAP_INV"
    assert repr(css.BSWAP**1.23) == "(css.BSWAP**1.23)"
    assert repr(shifted_bswap) == "css.BSwapPowGate(exponent=1.0, global_shift=0.3)"

    assert str(css.BSWAP) == str(shifted_bswap) == "BSWAP"
    assert str(css.BSWAP**-1) == str(shifted_bswap**-1) == "BSWAP_INV"
    assert str(css.BSWAP**1.23) == str(shifted_bswap**1.23) == "BSWAP**1.23"

    # Should iSWAP the (1, 1) and (2, 2) states:
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    assert cirq.Circuit(css.BSWAP(q0, q1)).final_state_vector(initial_state=8)[4] == 1j
    assert cirq.Circuit(css.BSWAP(q0, q1)).final_state_vector(initial_state=4)[8] == 1j
    assert cirq.Circuit(css.BSWAP_INV(q0, q1)).final_state_vector(initial_state=8)[4] == -1j
    assert cirq.Circuit(css.BSWAP_INV(q0, q1)).final_state_vector(initial_state=4)[8] == -1j

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.BSWAP(q0, q1)),
        textwrap.dedent(
            """
            0 (d=3): ───BSwap───
                        │
            1 (d=3): ───BSwap───
            """
        ),
    )

    cirq.testing.assert_json_roundtrip_works(
        css.BSWAP,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )
    cirq.testing.assert_json_roundtrip_works(
        shifted_bswap**1.23,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )


def test_qutrit_cz_pow_gate() -> None:
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        css.QutritCZPowGate,
        setup_code="import cirq_superstaq as css, sympy",
        ignore_decompose_to_default_gateset=True,
    )

    shifted_cz3 = css.QutritCZPowGate(global_shift=0.3)

    assert css.CZ3 == css.QutritCZPowGate()
    assert css.CZ3_INV == css.CZ3**-1 == css.CZ3**2

    assert css.CZ3 != shifted_cz3
    assert not cirq.approx_eq(css.CZ3, shifted_cz3)
    assert cirq.equal_up_to_global_phase(css.CZ3, shifted_cz3)
    assert cirq.equal_up_to_global_phase(css.CZ3**1.23, shifted_cz3**4.23)
    assert not cirq.equal_up_to_global_phase(css.CZ3, css.BSWAP)

    assert repr(css.CZ3) == "css.CZ3"
    assert repr(css.CZ3**-1) == "css.CZ3_INV"
    assert repr(css.CZ3**1.23) == "(css.CZ3**1.23)"
    assert repr(shifted_cz3) == "css.QutritCZPowGate(exponent=1.0, global_shift=0.3)"

    assert str(css.CZ3) == str(shifted_cz3) == "CZ3"
    assert str(css.CZ3**-1) == str(shifted_cz3**-1) == "CZ3_INV"
    assert str(css.CZ3**1.23) == str(shifted_cz3**1.23) == "CZ3**1.23"

    w = np.exp(2j * np.pi / 3)
    np.testing.assert_allclose(
        cirq.unitary(css.CZ3), np.diag(w ** np.array([0, 0, 0, 0, 1, 2, 0, 2, 4]))
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.CZ3.on(cirq.LineQid(1, 3), cirq.LineQid(2, 3))),
        textwrap.dedent(
            """
            1 (d=3): ───@───
                        │
            2 (d=3): ───@───
            """
        ),
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.CZ3_INV.on(cirq.LineQid(1, 3), cirq.LineQid(2, 3))),
        textwrap.dedent(
            """
            1 (d=3): ───@──────
                        │
            2 (d=3): ───@^-1───
            """
        ),
    )

    cirq.testing.assert_json_roundtrip_works(
        css.CZ3,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )
    cirq.testing.assert_json_roundtrip_works(
        shifted_cz3**1.23,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )


@pytest.mark.parametrize("dimension", [2, 3, 4, 5, 6])
def test_qutrit_cz_pow_gate_implementation_for_other_qudits(dimension: int) -> None:
    """Confirm that QutritCZPowGate._eigen_components_() would work for any dimension."""
    with mock.patch("cirq_superstaq.QutritCZPowGate.dimension", dimension):
        gate = css.QutritCZPowGate()
        assert gate.dimension == dimension
        assert cirq.qid_shape(gate) == (dimension, dimension)
        assert gate._period() == dimension
        assert cirq.equal_up_to_global_phase(gate**1.3, gate ** (dimension + 1.3))

        for exponent in [-3.7, -1.0, 0.0, 0.1, 1.0, 2.3, dimension]:
            np.testing.assert_allclose(
                scipy.linalg.fractional_matrix_power(cirq.unitary(gate), exponent),
                cirq.unitary(gate**exponent),
            )


def test_qutrit_z_pow_gate() -> None:
    assert not cirq.equal_up_to_global_phase(css.QutritZ0, css.QutritZ1)
    assert not cirq.equal_up_to_global_phase(css.QutritZ0, css.QutritZ2)
    assert not cirq.equal_up_to_global_phase(css.QutritZ1, css.QutritZ2)

    assert not cirq.approx_eq(css.QutritZ0, css.QutritZ1)
    assert not cirq.approx_eq(css.QutritZ0, css.QutritZ2)
    assert not cirq.approx_eq(css.QutritZ1, css.QutritZ2)

    assert repr(css.QutritZ0**1.0) == "css.QutritZ0"
    assert repr(css.QutritZ1**1.2) == "(css.QutritZ1**1.2)"
    assert (
        repr(css.QutritZ2PowGate(global_shift=0.5))
        == "css.QutritZ2PowGate(exponent=1.0, global_shift=0.5)"
    )

    assert str(css.QutritZ0**1.0) == "QutritZ0"
    assert str(css.QutritZ1**1.2) == "QutritZ1**1.2"
    assert str(css.QutritZ2PowGate(global_shift=0.5)) == "QutritZ2"

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QutritZ0(cirq.LineQid(0, 3))),
        "0 (d=3): ───Z₀───",
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QutritZ1(cirq.LineQid(0, 3)) ** 1.2),
        "0 (d=3): ───Z₁^-0.8───",
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QutritZ2PowGate(global_shift=0.5).on(cirq.LineQid(0, 3))),
        "0 (d=3): ---Z[2]---",
        use_unicode_characters=False,
    )


@pytest.mark.parametrize(
    "gate_type", [css.QutritZ0PowGate, css.QutritZ1PowGate, css.QutritZ2PowGate]
)
def test_qutrit_z_pow_gate_protocols(gate_type: Type[css.ops.qudit_gates._QutritZPowGate]) -> None:
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        gate_type,
        setup_code="import cirq_superstaq as css, sympy",
        ignore_decompose_to_default_gateset=True,
    )

    gate = gate_type(exponent=1.23)
    same_gate = gate_type() ** (gate.exponent + 2)
    similar_gate = gate_type(exponent=gate.exponent + 1e-10)
    shifted_gate = gate_type(exponent=gate.exponent, global_shift=0.5)
    another_gate = cirq.Y

    assert gate == same_gate
    assert gate != similar_gate
    assert gate != shifted_gate
    assert gate != another_gate

    assert cirq.approx_eq(gate, same_gate)
    assert cirq.approx_eq(gate, similar_gate)
    assert not cirq.approx_eq(gate, shifted_gate)
    assert not cirq.approx_eq(gate, another_gate)

    assert cirq.equal_up_to_global_phase(gate, same_gate)
    assert cirq.equal_up_to_global_phase(gate, similar_gate)
    assert cirq.equal_up_to_global_phase(gate, shifted_gate)
    assert not cirq.equal_up_to_global_phase(gate, another_gate)

    expected_unitary = np.eye(3, dtype=complex)
    expected_unitary[gate._target_state, gate._target_state] = np.exp(1.23j * np.pi)

    np.testing.assert_allclose(cirq.unitary(gate), expected_unitary)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(shifted_gate), expected_unitary, atol=1e-10
    )

    cirq.testing.assert_json_roundtrip_works(
        gate,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )
    cirq.testing.assert_json_roundtrip_works(
        shifted_gate**1.23,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )


def test_virtual_z_pow_gate() -> None:
    fixed_gates = [
        css.VirtualZPowGate(dimension=2, level=1),
        css.VirtualZPowGate(dimension=3, level=1),
        css.VirtualZPowGate(dimension=3, level=2),
        css.VirtualZPowGate(dimension=4, level=1),
        css.VirtualZPowGate(dimension=4, level=2),
        css.VirtualZPowGate(dimension=4, level=3),
    ]
    for i, gate in enumerate(fixed_gates):
        for other_gate in fixed_gates[:i]:
            assert not cirq.equal_up_to_global_phase(gate, other_gate)
            assert not cirq.approx_eq(gate, other_gate)

    assert repr(css.VirtualZPowGate(dimension=3, level=1)) == "css.VirtualZPowGate(dimension=3)"
    assert (
        repr(css.VirtualZPowGate(dimension=4, level=3))
        == "css.VirtualZPowGate(dimension=4, level=3)"
    )
    assert (
        repr(css.VirtualZPowGate(dimension=3, global_shift=0.5))
        == "css.VirtualZPowGate(dimension=3, global_shift=0.5)"
    )
    assert (
        repr(css.VirtualZPowGate(dimension=5, global_shift=0.5) ** 1.23)
        == "css.VirtualZPowGate(dimension=5, exponent=1.23, global_shift=0.5)"
    )

    assert str(css.VirtualZPowGate(dimension=2)) == "VZ(1+)"
    assert str(css.VirtualZPowGate(dimension=3)) == "VZ(1+)"
    assert str(css.VirtualZPowGate(dimension=3, level=2)) == "VZ(2+)"
    assert str(css.VirtualZPowGate(dimension=4, level=2) ** 1.2) == "VZ(2+)**1.2"
    assert str(css.VirtualZPowGate(dimension=4, global_shift=0.5)) == "VZ(1+)"
    assert str(css.VirtualZPowGate(dimension=5, global_shift=0.5) ** 2.3) == "VZ(1+)**2.3"

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.VirtualZPowGate(dimension=3, level=1)(cirq.LineQid(0, 3))),
        "0 (d=3): ───VZ₁₊───",
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.VirtualZPowGate(dimension=4, level=3)(cirq.LineQid(0, 4)) ** 1.2),
        "0 (d=4): ───VZ₃₊^-0.8───",
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.VirtualZPowGate(dimension=4, level=2)(cirq.LineQid(0, 4))),
        "0 (d=4): ---VZ(2+)---",
        use_unicode_characters=False,
    )

    with pytest.raises(ValueError, match="Invalid dimension"):
        _ = css.VirtualZPowGate(dimension=1)

    with pytest.raises(ValueError, match="Invalid energy level"):
        _ = css.VirtualZPowGate(dimension=3, level=3)

    with pytest.raises(ValueError, match="Invalid energy level"):
        _ = css.VirtualZPowGate(dimension=3, level=0)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_virtual_z_pow_gate_protocols(dimension: int, level: int) -> None:
    if level >= dimension:
        return

    gate_type = functools.partial(css.VirtualZPowGate, dimension=dimension, level=level)
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        cast("Type[cirq.EigenGate]", gate_type),
        setup_code="import cirq_superstaq as css, sympy",
        ignore_decompose_to_default_gateset=True,
    )

    gate = css.VirtualZPowGate(dimension, level=level, exponent=1.23)
    same_gate = css.VirtualZPowGate(dimension, level=level - dimension) ** (gate.exponent + 2)
    similar_gate = css.VirtualZPowGate(dimension, level=level, exponent=1.23 + 1e-10)
    shifted_gate = css.VirtualZPowGate(dimension, level=level, exponent=1.23, global_shift=0.5)
    another_gate = cirq.Y

    assert gate == same_gate
    assert gate != similar_gate
    assert gate != shifted_gate
    assert gate != another_gate

    assert cirq.approx_eq(gate, same_gate)
    assert cirq.approx_eq(gate, similar_gate)
    assert not cirq.approx_eq(gate, shifted_gate)
    assert not cirq.approx_eq(gate, another_gate)

    assert cirq.equal_up_to_global_phase(gate, same_gate)
    assert cirq.equal_up_to_global_phase(gate, similar_gate)
    assert cirq.equal_up_to_global_phase(gate, shifted_gate)
    assert not cirq.equal_up_to_global_phase(gate, another_gate)

    expected_unitary = np.eye(dimension, dtype=complex)
    expected_unitary[level:, level:] *= np.exp(1.23j * np.pi)

    np.testing.assert_allclose(cirq.unitary(gate), expected_unitary)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(shifted_gate), expected_unitary, atol=1e-10
    )

    cirq.testing.assert_json_roundtrip_works(
        gate,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )
    cirq.testing.assert_json_roundtrip_works(
        shifted_gate**1.23,
        resolvers=css.SUPERSTAQ_RESOLVERS,
    )


def test_qubit_subspace_gate() -> None:
    assert css.QubitSubspaceGate(cirq.X, (3,), [(0, 2)]) == css.QubitSubspaceGate(
        cirq.X, (3,), [(0, -1)]
    )
    with pytest.raises(ValueError, match="Only qubit gates"):
        _ = css.QubitSubspaceGate(cirq.ZPowGate(dimension=3), (3,))

    with pytest.raises(ValueError, match="same number of qubits"):
        _ = css.QubitSubspaceGate(cirq.Z, (3, 3))

    with pytest.raises(ValueError, match="Invalid qid_shape"):
        _ = css.QubitSubspaceGate(cirq.CZ, (1, 2))

    with pytest.raises(ValueError, match="two subspace indices for every qubit"):
        _ = css.QubitSubspaceGate(cirq.CZ, (3, 3), [(0, 1)])

    gate = css.QubitSubspaceGate(cirq.X, (3,))
    assert str(gate) == "QubitSubspaceGate(X, (3,))"

    gate = css.QubitSubspaceGate(cirq.X, (3,), [(0, 2)])
    assert str(gate) == "QubitSubspaceGate(X, (3,), [(0, 2)])"

    gate = css.QubitSubspaceGate(css.ZZSwapGate(1.23), (3, 3))
    assert gate**3 == css.QubitSubspaceGate(css.ZZSwapGate(3.69), (3, 3))
    with pytest.raises(TypeError):
        _ = gate**3.1


@pytest.mark.parametrize(
    "sub_gate_type, qid_shape, subspaces",
    [
        (cirq.XPowGate, (3,), None),
        (cirq.YPowGate, (3,), [(1, 2)]),
        (cirq.ZPowGate, (5,), [(1, 4)]),
        (cirq.ISwapPowGate, (3, 4), [(1, 2), (0, 3)]),
        (cirq.CCZPowGate, (3, 3, 2), [(1, 2), (0, 1), (1, 0)]),
    ],
)
def test_qubit_subspace_gate_protocols(
    sub_gate_type: Type[cirq.EigenGate],
    qid_shape: Tuple[int, ...],
    subspaces: Optional[List[Tuple[int, int]]],
) -> None:
    sub_gate = sub_gate_type(exponent=1.23, global_shift=0.0)

    gate = css.QubitSubspaceGate(sub_gate, qid_shape, subspaces)
    same_gate = css.QubitSubspaceGate(sub_gate._with_exponent(1.0), qid_shape, subspaces) ** 1.23
    similar_gate = css.QubitSubspaceGate(
        sub_gate._with_exponent(1.23 + 1e-10), qid_shape, subspaces
    )
    shifted_gate = css.QubitSubspaceGate(
        sub_gate_type(exponent=1.23, global_shift=0.5), qid_shape, subspaces
    )
    flipped_gate = css.QubitSubspaceGate(sub_gate, qid_shape, [(j, i) for i, j in gate.subspaces])
    larger_gate = css.QubitSubspaceGate(sub_gate, (8,) * len(qid_shape), subspaces)
    another_gate = cirq.Y

    assert gate == gate
    assert gate == same_gate
    assert gate != similar_gate
    assert gate != shifted_gate
    assert gate != flipped_gate
    assert gate != larger_gate
    assert gate != another_gate

    assert cirq.approx_eq(gate, gate)
    assert cirq.approx_eq(gate, same_gate)
    assert cirq.approx_eq(gate, similar_gate)
    assert not cirq.approx_eq(gate, shifted_gate)
    assert not cirq.approx_eq(gate, flipped_gate)
    assert not cirq.approx_eq(gate, larger_gate)
    assert not cirq.approx_eq(gate, another_gate)

    assert cirq.equal_up_to_global_phase(gate, gate)
    assert cirq.equal_up_to_global_phase(gate, same_gate)
    assert cirq.equal_up_to_global_phase(gate, similar_gate)
    assert not cirq.equal_up_to_global_phase(gate, shifted_gate)
    assert not cirq.equal_up_to_global_phase(gate, flipped_gate)
    assert not cirq.equal_up_to_global_phase(gate, larger_gate)
    assert not cirq.equal_up_to_global_phase(gate, another_gate)

    cirq.testing.assert_implements_consistent_protocols(
        gate,
        setup_code="import cirq, cirq_superstaq as css, sympy",
        ignore_decompose_to_default_gateset=True,
    )
    cirq.testing.assert_json_roundtrip_works(
        gate, resolvers=[*css.SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS]
    )

    # Check that it has the correct unitary in the correct subspace:
    n = cirq.num_qubits(gate)
    unitary = cirq.unitary(gate).reshape(2 * gate._qid_shape)
    for qi in range(n):
        subspace = gate._subspaces[qi]
        unitary = unitary.take(subspace, qi).take(subspace, qi + n)

    assert unitary.shape == (2, 2) * n
    np.testing.assert_array_equal(unitary.reshape(2**n, 2**n), cirq.unitary(gate.sub_gate))


def test_qubit_subspace_circuit_diagram() -> None:
    q0 = cirq.LineQid(0, dimension=3)
    q1 = cirq.LineQid(1, dimension=4)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QubitSubspaceGate(cirq.rx(np.pi / 2), (3,), [(0, 2)]).on(q0)),
        textwrap.dedent(
            """
            0 (d=3): ───Rx(0.5π)₀₂───
            """
        ),
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QubitSubspaceGate(cirq.CX, (3, 4), [(0, 1), (1, 2)]).on(q0, q1)),
        textwrap.dedent(
            """
            0 (d=3): ───@₀₁───
                        │
            1 (d=4): ───X₁₂───
            """
        ),
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.QubitSubspaceGate(cirq.rx(np.pi / 2), (4,), [(0, 2)]).on(q1)),
        textwrap.dedent(
            """
            1 (d=4): ---Rx(0.5pi)[0,2]---
            """
        ),
        use_unicode_characters=False,
    )


def test_qubit_subspace_op() -> None:
    assert css.qubit_subspace_op(
        cirq.CZ(cirq.LineQubit(2), cirq.LineQid(1, 2)), (3, 4)
    ) == css.QubitSubspaceGate(cirq.CZ, (3, 4)).on(cirq.LineQid(2, 3), cirq.LineQid(1, 4))
    assert css.qubit_subspace_op(
        cirq.Z(cirq.GridQubit(5, 9)) ** 1.23, (7,)
    ) == css.QubitSubspaceGate(cirq.Z**1.23, (7,)).on(cirq.GridQid(5, 9, dimension=7))
    assert css.qubit_subspace_op(
        cirq.X(cirq.NamedQubit("qubit")), (4,), [(1, -1)]
    ) == css.QubitSubspaceGate(cirq.X, (4,), [(1, 3)]).on(cirq.NamedQid("qubit", dimension=4))
    with pytest.raises(ValueError, match="has no gate."):
        _ = css.qubit_subspace_op(cirq.CircuitOperation(cirq.FrozenCircuit()), ())
