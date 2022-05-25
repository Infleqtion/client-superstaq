from typing import List, Set

import numpy as np
import pytest
import qiskit

import qiskit_superstaq as qss


def _check_gate_definition(gate: qiskit.circuit.Gate) -> None:
    """Check gate.definition, gate.__array__(), and gate.inverse() against one another"""

    assert np.allclose(gate.to_matrix(), gate.__array__())
    defined_operation = qiskit.quantum_info.Operator(gate.definition)
    assert defined_operation.is_unitary()
    assert defined_operation.equiv(gate.to_matrix(), atol=1e-10)

    inverse_operation = qiskit.quantum_info.Operator(gate.inverse().definition)
    assert inverse_operation.is_unitary()

    assert inverse_operation.equiv(gate.inverse().to_matrix(), atol=1e-10)
    assert inverse_operation.equiv(gate.to_matrix().T.conj(), atol=1e-10)


def test_acecr() -> None:
    gate = qss.AceCR("+-")
    _check_gate_definition(gate)
    assert repr(gate) == "qss.AceCR('+-')"
    assert str(gate) == "AceCR+-"
    assert gate.qasm() == "acecr_pm"

    gate = qss.AceCR("-+", label="label")
    _check_gate_definition(gate)
    assert repr(gate) == "qss.AceCR('-+', label='label')"
    assert str(gate) == "AceCR-+"
    assert gate.qasm() == "acecr_mp"

    gate = qss.AceCR("-+", sandwich_rx_rads=np.pi / 2)
    _check_gate_definition(gate)
    assert repr(gate) == "qss.AceCR('-+', sandwich_rx_rads=1.5707963267948966)"
    assert str(gate) == "AceCR-+|RXGate(pi/2)|"
    assert gate.qasm() == "acecr_mp_rx(pi/2)"

    gate = qss.AceCR("-+", sandwich_rx_rads=np.pi / 2, label="label")
    _check_gate_definition(gate)
    assert repr(gate) == "qss.AceCR('-+', sandwich_rx_rads=1.5707963267948966, label='label')"
    assert str(gate) == "AceCR-+|RXGate(pi/2)|"
    assert gate.qasm() == "acecr_mp_rx(pi/2)"

    with pytest.raises(ValueError, match="Polarity must be"):
        _ = qss.AceCR("++")


def test_zz_swap() -> None:
    gate = qss.ZZSwapGate(1.23)
    _check_gate_definition(gate)
    assert repr(gate) == "qss.ZZSwapGate(1.23)"
    assert str(gate) == "ZZSwapGate(1.23)"

    gate = qss.ZZSwapGate(4.56, label="label")
    assert repr(gate) == "qss.ZZSwapGate(4.56, label='label')"
    assert str(gate) == "ZZSwapGate(4.56)"


def test_parallel_gates() -> None:
    gate = qss.ParallelGates(
        qss.AceCR("+-"),
        qiskit.circuit.library.RXGate(1.23),
    )
    assert str(gate) == "ParallelGates(acecr_pm, rx(1.23))"
    _check_gate_definition(gate)

    # confirm gates are applied to disjoint qubits
    all_qargs: Set[qiskit.circuit.Qubit] = set()
    for _, qargs, _ in gate.definition:
        assert all_qargs.isdisjoint(qargs)
        all_qargs.update(qargs)
    assert len(all_qargs) == gate.num_qubits

    # double check qubit ordering
    qc1 = qiskit.QuantumCircuit(3)
    qc1.append(gate, [0, 2, 1])

    qc2 = qiskit.QuantumCircuit(3)
    qc2.rx(1.23, 1)
    qc2.append(qss.AceCR("+-"), [0, 2])

    assert qiskit.quantum_info.Operator(qc1).equiv(qc2, atol=1e-14)

    gate = qss.ParallelGates(
        qiskit.circuit.library.XGate(),
        qss.ZZSwapGate(1.23),
        qiskit.circuit.library.ZGate(),
        label="label",
    )
    assert str(gate) == "ParallelGates(x, zzswap(1.23), z)"
    _check_gate_definition(gate)

    # confirm gates are applied to disjoint qubits
    all_qargs.clear()
    for _, qargs, _ in gate.definition:
        assert all_qargs.isdisjoint(qargs)
        all_qargs.update(qargs)
    assert len(all_qargs) == gate.num_qubits

    gate = qss.ParallelGates(
        qiskit.circuit.library.XGate(),
        qss.ParallelGates(
            qiskit.circuit.library.YGate(),
            qiskit.circuit.library.ZGate(),
        ),
    )
    gate2 = qss.ParallelGates(
        qiskit.circuit.library.XGate(),
        qiskit.circuit.library.YGate(),
        qiskit.circuit.library.ZGate(),
    )
    assert gate.component_gates == gate2.component_gates
    assert gate == gate2

    with pytest.raises(ValueError, match="Component gates must be"):
        _ = qss.ParallelGates(qiskit.circuit.Measure())


def test_ix_gate() -> None:
    gate = qss.custom_gates.iXGate()
    _check_gate_definition(gate)
    assert repr(gate) == "qss.custom_gates.iXGate(label=None)"
    assert str(gate) == "iXGate(label=None)"

    assert gate.inverse() == qss.custom_gates.iXdgGate()
    assert gate.control(2) == qss.custom_gates.iCCXGate()
    assert type(gate.control(1)) is qiskit.circuit.ControlledGate
    assert np.all(gate.to_matrix() == [[0, 1j], [1j, 0]])


def test_ixdg_gate() -> None:
    gate = qss.custom_gates.iXdgGate()
    _check_gate_definition(gate)
    assert repr(gate) == "qss.custom_gates.iXdgGate(label=None)"
    assert str(gate) == "iXdgGate(label=None)"

    assert gate.inverse() == qss.custom_gates.iXGate()
    assert gate.control(2) == qss.custom_gates.iCCXdgGate()
    assert type(gate.control(1)) is qiskit.circuit.ControlledGate
    assert np.all(gate.to_matrix() == [[0, -1j], [-1j, 0]])


def test_iccx() -> None:
    gate = qss.custom_gates.iCCXGate()
    _check_gate_definition(gate)
    assert repr(gate) == "qss.custom_gates.iCCXGate(label=None, ctrl_state=3)"
    assert str(gate) == "iCCXGate(label=None, ctrl_state=3)"


def test_iccxdg() -> None:
    gate = qss.custom_gates.iCCXdgGate()
    _check_gate_definition(gate)
    assert repr(gate) == "qss.custom_gates.iCCXdgGate(label=None, ctrl_state=3)"
    assert str(gate) == "iCCXdgGate(label=None, ctrl_state=3)"


def test_aqticcx() -> None:
    gate = qss.AQTiCCXGate()
    _check_gate_definition(gate)

    assert repr(gate) == "qss.custom_gates.iCCXGate(label=None, ctrl_state=0)"
    assert str(gate) == "iCCXGate(label=None, ctrl_state=0)"

    qc = qiskit.QuantumCircuit(3)

    qc.append(qss.AQTiCCXGate(), [0, 1, 2])

    correct_unitary = np.array(
        [
            [0, 0, 0, 0, 1j, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1j, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
    )

    np.allclose(qiskit.quantum_info.Operator(qc), correct_unitary)


def test_custom_resolver() -> None:
    custom_gates: List[qiskit.circuit.Gate] = [
        qss.AceCR("+-"),
        qss.AceCR("-+"),
        qss.AceCR("+-", 1.23),
        qss.ZZSwapGate(1.23),
        qss.AQTiCCXGate(),
        qss.custom_gates.iXGate(),
        qss.custom_gates.iXdgGate(),
        qss.custom_gates.iCCXGate(),
        qss.custom_gates.iCCXGate(ctrl_state="01"),
        qss.custom_gates.iCCXGate(ctrl_state="10"),
        qss.custom_gates.iCCXdgGate(ctrl_state="00"),
        qss.custom_gates.iCCXdgGate(),
        qss.custom_gates.iCCXdgGate(ctrl_state="01"),
        qss.custom_gates.iCCXdgGate(ctrl_state="10"),
    ]

    generic_gates = []

    for custom_gate in custom_gates:
        generic_gate = qiskit.circuit.Gate(
            custom_gate.name,
            custom_gate.num_qubits,
            custom_gate.params,
            label=str(id(custom_gate)),
        )
        generic_gate.definition = custom_gate.definition
        generic_gates.append(generic_gate)
        assert generic_gate != custom_gate

        resolved_gate = qss.custom_gates.custom_resolver(generic_gate)
        assert resolved_gate == custom_gate
        assert resolved_gate.label == str(id(custom_gate))  # (labels aren't included in __eq__)

    parallel_gates = qss.ParallelGates(
        qiskit.circuit.library.RXGate(4.56), qiskit.circuit.library.CXGate(), *custom_gates
    )
    parallel_generic_gates = qss.ParallelGates(
        qiskit.circuit.library.RXGate(4.56),
        qiskit.circuit.library.CXGate(),
        *generic_gates,
        label="label-1"
    )
    resolved_gate = qss.custom_gates.custom_resolver(parallel_generic_gates)
    assert parallel_generic_gates != parallel_gates
    assert resolved_gate == parallel_gates
    assert resolved_gate.label == "label-1"

    generic_parallel_gates = qiskit.circuit.Gate(
        parallel_gates.name, parallel_gates.num_qubits, [], label="label-2"
    )
    generic_parallel_gates.definition = parallel_generic_gates.definition
    resolved_gate = qss.custom_gates.custom_resolver(generic_parallel_gates)
    assert generic_parallel_gates != parallel_gates
    assert resolved_gate == parallel_gates
    assert resolved_gate.label == "label-2"

    assert qss.custom_gates.custom_resolver(qiskit.circuit.library.CXGate()) is None
    assert qss.custom_gates.custom_resolver(qiskit.circuit.library.RXGate(2)) is None
    assert qss.custom_gates.custom_resolver(qiskit.circuit.Gate("??", 1, [])) is None
