import itertools
import textwrap

import cirq
import numpy as np
import packaging
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

    assert gate**1 == gate
    assert gate**0 == css.ZZSwapGate(0.0)
    assert gate**-1 == css.ZZSwapGate(-0.123)

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
    css.AceCR("-+", np.pi / 3)
    with pytest.raises(ValueError, match="Polarity must be"):
        css.AceCR("++")


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

    circuit = cirq.Circuit(css.AceCR("+-", np.pi)(*qubits))
    assert str(circuit) == textwrap.dedent(
        """\
        0: ───AceCR+-(Z side)──────────
              │
        1: ───AceCR+-(X side)|Rx(π)|───"""
    )


def test_acecr_qasm() -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        css.AceCR("+-").on(*qubits),
        css.AceCR("-+").on(*reversed(qubits)),
        css.AceCR("-+", np.pi / 2).on(*qubits),
    )

    assert circuit.to_qasm(header="") == textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";


        // Qubits: [q(0), q(1)]
        qreg q[2];


        acecr_pm q[0],q[1];
        acecr_mp q[1],q[0];
        acecr_mp_rx(pi*0.5) q[0],q[1];
        """
    )


def test_acecr_eq() -> None:
    assert css.AceCRPlusMinus == css.AceCR("+-")
    assert css.AceCRPlusMinus != css.AceCR("-+")
    assert css.AceCR("+-", np.pi) == css.AceCR("+-", np.pi)
    assert css.AceCR("-+", np.pi) != css.AceCR("+-", np.pi)

    assert css.AceCR("+-", np.pi) == css.AceCR("+-", 5 * np.pi)
    assert css.AceCR("+-", np.pi) == css.AceCR("+-", 3 * np.pi)

    assert cirq.approx_eq(css.AceCR("+-", np.pi), css.AceCR("+-", -np.pi))
    assert cirq.approx_eq(css.AceCR("+-", np.pi), css.AceCR("+-", 3 * np.pi))


def test_acecr_repr_and_str() -> None:
    assert repr(css.AceCRMinusPlus) == "css.AceCR('-+')"
    assert repr(css.AceCR("+-", np.pi)) == "css.AceCR('+-', 3.141592653589793)"
    cirq.testing.assert_equivalent_repr(
        css.AceCRMinusPlus, setup_code="import cirq_superstaq as css"
    )
    cirq.testing.assert_equivalent_repr(
        css.AceCR("+-", np.pi), setup_code="import cirq; import cirq_superstaq as css"
    )
    assert str(css.AceCRMinusPlus) == "AceCR-+"
    assert str(css.AceCR("+-", np.pi)) == "AceCR+-|Rx(π)|"


def test_acecr_decompose() -> None:
    a = cirq.LineQubit(0)
    b = cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.decompose_once(css.AceCRMinusPlus(a, b)))
    assert len(circuit) == 3 and len(list(circuit.all_operations())) == 3

    circuit = cirq.Circuit(cirq.decompose_once(css.AceCR("+-", -np.pi / 2)(a, b)))
    assert len(circuit) == 3 and len(list(circuit.all_operations())) == 4


def test_barrier() -> None:
    n = 3
    qubits = cirq.LineQubit.range(n)
    gate = css.Barrier(n)

    assert str(gate) == "Barrier(3)"
    assert repr(gate) == "css.Barrier(3)"

    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_superstaq as css")

    operation = gate.on(*qubits)
    assert cirq.decompose(operation) == [operation]

    # confirm Barrier is as an InterchangeableQubitsGate
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


def test_parallel_gates_circuit_diagram_fallback() -> None:
    gate = cirq.circuits.qasm_output.QasmUGate(0.1, 0.2, 0.3)
    assert not hasattr(gate, "_circuit_diagram_info_")

    cirq.testing.assert_has_diagram(
        cirq.Circuit(css.ParallelGates(gate).on(cirq.LineQubit(1))),
        f"1: ───ParallelGates({gate})───",
    )


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
        else:
            assert operation != gate(*permuted_qubits)

    with pytest.raises(ValueError, match="index out of range"):
        _ = gate.qubit_index_to_equivalence_group_key(4)

    with pytest.raises(ValueError, match="index out of range"):
        _ = gate.qubit_index_to_equivalence_group_key(-1)


@pytest.mark.skipif(
    packaging.version.parse("0.14.0.dev20220126174724")
    < packaging.version.parse(cirq.__version__)
    < packaging.version.parse("0.15.0.dev20220420201205"),
    reason="https://github.com/quantumlib/Cirq/issues/5148",
)
def test_parallel_gates_equivalence_groups_nonadjacent() -> None:  # pragma: no cover
    """Fails in cirq version 0.14.x due to https://github.com/quantumlib/Cirq/issues/5148"""
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
        else:
            assert operation != gate(*permuted_qubits)


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


def test_parallel_rgate() -> None:
    qubits = cirq.LineQubit.range(2)

    rot_gate = css.ParallelRGate(1.23 * np.pi, 4.56 * np.pi, len(qubits))
    cirq.testing.assert_equivalent_repr(
        rot_gate, setup_code="import cirq; import cirq_superstaq as css"
    )
    text = f"RGate({rot_gate.phase_exponent}π, {rot_gate.exponent}π) x {len(qubits)}"
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


def test_ixgate() -> None:
    gate = css.custom_gates.IXGate()

    assert str(gate) == "IX"
    assert repr(gate) == "css.custom_gates.IX"
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

    assert isinstance(gate**1, css.custom_gates.IXGate)
    assert isinstance(gate**5, css.custom_gates.IXGate)
    assert isinstance(gate**1.5, cirq.XPowGate)
    assert not isinstance(gate**1.5, css.custom_gates.IXGate)

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
        cirq.unitary(css.custom_gates.ICCX(*qubits)),
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
    circuit += css.AceCR("+-", -np.pi / 2)(qubits[0], qubits[1])
    circuit += css.ParallelGates(cirq.X, css.ZX).on(qubits[0], qubits[2], qubits[3])
    circuit += cirq.ms(1.23).on(qubits[0], qubits[1])
    circuit += css.RGate(1.23, 4.56).on(qubits[0])
    circuit += css.ParallelRGate(1.23, 4.56, len(qubits)).on(*qubits)
    circuit += css.AQTITOFFOLI(qubits[0], qubits[1], qubits[2])
    circuit += css.custom_gates.ICCX(qubits[0], qubits[1], qubits[2])
    circuit += css.custom_gates.IX(qubits[0])
    circuit += cirq.CX(qubits[0], qubits[1])

    json_text = cirq.to_json(circuit)
    resolvers = [css.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    assert cirq.read_json(json_text=json_text, resolvers=resolvers) == circuit
