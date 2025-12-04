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
from __future__ import annotations

import numpy as np
import pytest
import qiskit

import qiskit_superstaq as qss


def _check_gate_definition(gate: qiskit.circuit.Gate) -> None:
    """Check gate.definition, gate.__array__(), and gate.inverse() against one another."""
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
    assert repr(gate) == "qss.AceCR()"
    assert str(gate) == "AceCR"

    gate = qss.AceCR("-+", label="label")
    assert repr(gate) == "qss.AceCR(rads=-1.5707963267948966, label='label')"
    assert str(gate) == "AceCR(-0.5π)"

    gate = qss.AceCR("-+", sandwich_rx_rads=np.pi / 2)
    assert repr(gate) == "qss.AceCR(rads=-1.5707963267948966, sandwich_rx_rads=1.5707963267948966)"
    assert str(gate) == "AceCR(-0.5π)|RXGate(π/2)|"

    gate = qss.AceCR("-+", sandwich_rx_rads=np.pi / 2, label="label")
    _check_gate_definition(gate)
    assert (
        repr(gate)
        == "qss.AceCR(rads=-1.5707963267948966, sandwich_rx_rads=1.5707963267948966, label='label')"
    )
    assert str(gate) == "AceCR(-0.5π)|RXGate(π/2)|"

    with pytest.raises(ValueError, match=r"Polarity must be"):
        _ = qss.AceCR("++")

    gate = qss.AceCR(np.pi)
    assert repr(gate) == "qss.AceCR(rads=3.141592653589793)"
    assert str(gate) == "AceCR(1.0π)"

    gate = qss.AceCR(sandwich_rx_rads=np.pi / 2)
    assert repr(gate) == "qss.AceCR(sandwich_rx_rads=1.5707963267948966)"
    assert str(gate) == "AceCR|RXGate(π/2)|"

    gate = qss.AceCR(rads=np.pi / 5, sandwich_rx_rads=np.pi / 2)
    assert repr(gate) == "qss.AceCR(rads=0.6283185307179586, sandwich_rx_rads=1.5707963267948966)"
    assert str(gate) == "AceCR(0.2π)|RXGate(π/2)|"

    qc = qiskit.QuantumCircuit(2)
    qc.append(gate, [0, 1])
    correct_unitary = np.array(
        [
            [0, 0.891007, 0, -0.45399j],
            [0.45399, 0j, -0.891007j, 0],
            [0, -0.45399j, 0, 0.891007],
            [-0.891007j, 0, 0.45399, 0],
        ],
    )
    np.testing.assert_allclose(gate, correct_unitary, atol=1e-6)
    np.testing.assert_allclose(qiskit.quantum_info.Operator(qc), correct_unitary, atol=1e-6)

    with pytest.raises(ValueError, match=r"without making a copy"):
        _ = gate.__array__(copy=False)

    np.testing.assert_array_equal(
        np.array(gate, dtype=np.complex64), gate.__array__(dtype=np.complex64), strict=True
    )


def test_zz_swap() -> None:
    gate = qss.ZZSwapGate(1.23)
    _check_gate_definition(gate)
    assert repr(gate) == "qss.ZZSwapGate(1.23)"
    assert str(gate) == "ZZSwapGate(1.23)"

    gate = qss.ZZSwapGate(4.56, label="label")
    assert repr(gate) == "qss.ZZSwapGate(4.56, label='label')"
    assert str(gate) == "ZZSwapGate(4.56)"

    gate = qss.ZZSwapGate(np.pi / 3)
    assert str(gate) == "ZZSwapGate(π/3)"

    with pytest.raises(ValueError, match=r"without making a copy"):
        _ = gate.__array__(copy=False)

    np.testing.assert_array_equal(
        np.array(gate, dtype=np.complex64), gate.__array__(dtype=np.complex64), strict=True
    )


def test_stripped_cz() -> None:
    gate = qss.StrippedCZGate(1.23)
    _check_gate_definition(gate)
    assert repr(gate) == "qss.StrippedCZGate(1.23)"
    assert str(gate) == "StrippedCZGate(1.23)"

    gate = qss.StrippedCZGate(np.pi / 3)
    assert str(gate) == "StrippedCZGate(π/3)"

    with pytest.raises(ValueError, match=r"without making a copy"):
        _ = gate.__array__(copy=False)

    np.testing.assert_array_equal(
        np.array(gate, dtype=np.complex64), gate.__array__(dtype=np.complex64), strict=True
    )


def test_parallel_gates() -> None:
    gate = qss.ParallelGates(
        qss.AceCR("+-"),
        qiskit.circuit.library.RXGate(1.23),
    )
    assert str(gate) == "ParallelGates(acecr(π/2), rx(1.23))"
    _check_gate_definition(gate)

    # confirm gates are applied to disjoint qubits
    all_qargs: set[qiskit.circuit.Qubit] = set()
    for inst in gate.definition:
        assert all_qargs.isdisjoint(inst.qubits)
        all_qargs.update(inst.qubits)
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
    for inst in gate.definition:
        assert all_qargs.isdisjoint(inst.qubits)
        all_qargs.update(inst.qubits)
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

    with pytest.raises(TypeError, match=r"Component gates must be"):
        _ = qss.ParallelGates(qiskit.circuit.Measure())

    with pytest.raises(ValueError, match=r"without making a copy"):
        _ = gate.__array__(copy=False)

    np.testing.assert_array_equal(
        np.array(gate, dtype=np.complex64), gate.__array__(dtype=np.complex64), strict=True
    )


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


def test_dd_gate() -> None:
    gate = qss.DDGate(1.23)
    _check_gate_definition(gate)
    assert repr(gate) == "qss.DDGate(1.23)"
    assert str(gate) == "DDGate(1.23)"

    gate = qss.DDGate(4.56, label="label")
    assert repr(gate) == "qss.DDGate(4.56, label='label')"
    assert str(gate) == "DDGate(4.56)"

    gate = qss.DDGate(np.pi / 3)
    assert str(gate) == "DDGate(π/3)"

    with pytest.raises(ValueError, match=r"without making a copy"):
        _ = gate.__array__(copy=False)

    np.testing.assert_array_equal(
        np.array(gate, dtype=np.complex64), gate.__array__(dtype=np.complex64), strict=True
    )
