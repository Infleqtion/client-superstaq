import textwrap
from unittest import mock

import cirq
import numpy as np
import pytest
import scipy.linalg

import cirq_superstaq as css


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

    # Should iSWAP the (0, 2) and (1, 1) states:
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    assert cirq.Circuit(css.BSWAP(q0, q1)).final_state_vector(initial_state=2)[4] == 1j
    assert cirq.Circuit(css.BSWAP(q0, q1)).final_state_vector(initial_state=4)[2] == 1j
    assert cirq.Circuit(css.BSWAP_INV(q0, q1)).final_state_vector(initial_state=2)[4] == -1j
    assert cirq.Circuit(css.BSWAP_INV(q0, q1)).final_state_vector(initial_state=4)[2] == -1j

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
