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

import inspect
import re
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cirq
import cirq_superstaq as css
import general_superstaq as gss
import numpy as np
import pytest
import qiskit
import qiskit_superstaq as qss
import sympy

import supermarq as sm

if TYPE_CHECKING:
    import _pytest


def parametrize_over_qiskit_gates(
    *trial_gates: qiskit.circuit.Instruction,
) -> _pytest.mark.structures.MarkDecorator:
    ids = [f"{gate.base_class.__name__}:{gate.name}" for gate in trial_gates]
    return pytest.mark.parametrize("trial_gate", trial_gates, ids=ids)


def random_qubits(num_qubits: int) -> list[cirq.Qid]:
    rng = np.random.default_rng()
    indices = rng.choice(num_qubits + 50, size=num_qubits, replace=False)
    return [cirq.LineQubit(int(idx)) for idx in indices]


def _operations_are_equivalent(
    qiskit_inst: qiskit.circuit.Instruction | qiskit.QuantumCircuit,
    cirq_op: cirq.Operation | cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    phase: float = 0,
) -> bool:
    """Check that a qiskit gate (or QuantumCircuit) has the same unitary up to exp(1j * phase)
    as a cirq.Operation or cirq.Circuit applied to qubits in a specified order.
    """
    cirq_circuit = cirq.Circuit(cirq_op)
    assert set(qubit_order).issuperset(cirq_circuit.all_qubits())
    assert len(qubit_order) == qiskit_inst.num_qubits

    # qubit_order reversed to convert between cirq-endian and qiskit-endian
    cirq_unitary = cirq_circuit.unitary(reversed(qubit_order))
    qiskit_unitary = qiskit.quantum_info.Operator(qiskit_inst).data

    return np.allclose(np.exp(1j * phase) * cirq_unitary, qiskit_unitary)


def _gates_are_equivalent(
    qiskit_inst: qiskit.circuit.Instruction,
    cirq_gate: cirq.Gate,
    ignore_global_phase: bool = False,
) -> bool:
    """Check that a qiskit gate and cirq gate have the same unitary, up to exp(1j * phase)."""
    qubits = cirq.LineQubit.range(cirq.num_qubits(cirq_gate))
    assert len(qubits) == qiskit_inst.num_qubits

    # qubits reversed to convert between cirq-endian and qiskit-endian
    cirq_unitary = cirq.Circuit(cirq_gate.on(*qubits)).unitary(reversed(qubits))
    qiskit_unitary = qiskit.quantum_info.Operator(qiskit_inst).data
    if ignore_global_phase:
        return cirq.allclose_up_to_global_phase(cirq_unitary, qiskit_unitary)

    return np.allclose(cirq_unitary, qiskit_unitary)


def trial_qiskit_static_gates() -> list[qiskit.circuit.Gate]:
    return [qiskit_gate_type() for qiskit_gate_type in sm.converters.qiskit._qiskit_static_gates]


def trial_qiskit_pow_gates() -> list[qiskit.circuit.Gate]:
    rng = np.random.default_rng()
    qiskit_gates = []
    for qiskit_gate_type in sm.converters.qiskit._qiskit_pow_gates:
        rads = rng.uniform(-2 * np.pi, 2 * np.pi)
        qiskit_gate = qiskit_gate_type(rads)
        qiskit_gates.append(qiskit_gate)

        assert _gates_are_equivalent(
            qiskit_gate, sm.converters.qiskit._qiskit_pow_gates[qiskit_gate_type] ** (rads / np.pi)
        )

    return qiskit_gates


def trial_qiskit_param_gates() -> list[qiskit.circuit.Gate]:
    rng = np.random.default_rng()
    qiskit_gates = [
        qiskit.circuit.Barrier(3),
        qiskit.circuit.Delay(rng.uniform(0, 10), unit="ps"),
        qiskit.circuit.Delay(rng.uniform(0, 10), unit="ns"),
        qiskit.circuit.Delay(rng.uniform(0, 10), unit="us"),
        qiskit.circuit.Delay(rng.uniform(0, 10), unit="ms"),
        qiskit.circuit.Delay(rng.uniform(0, 10), unit="s"),
        qss.AceCR("-+"),
        qss.AceCR(rng.uniform(-1 * np.pi, 1 * np.pi)),
        qss.AceCR("+-", sandwich_rx_rads=rng.uniform(-1 * np.pi, 1 * np.pi)),
        qss.AceCR("-+", sandwich_rx_rads=rng.uniform(-1 * np.pi, 1 * np.pi)),
        qss.AceCR(np.pi),
        qiskit.circuit.library.RGate(*rng.uniform(-2 * np.pi, 2 * np.pi, size=2)),
        qss.ZZSwapGate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qss.StrippedCZGate(rng.uniform(0, 2 * np.pi)),
        qiskit.circuit.library.PermutationGate(rng.permutation(4)),
    ]

    # meta: make sure we've covered every gate type in _qiskit_param_gates
    covered = {gate.base_class for gate in qiskit_gates}
    missing = set(sm.converters.qiskit._qiskit_param_gates).difference(covered)
    assert not missing, "Missing tests for: " + ", ".join(gate.__name__ for gate in missing)
    return qiskit_gates


def trial_controlled_gates() -> list[qiskit.circuit.ControlledGate]:
    # An example of every gate in _known_qiskit_controlled_gate_types
    rng = np.random.default_rng()
    qiskit_gates = [
        qiskit.circuit.library.CXGate(),
        qiskit.circuit.library.CYGate(),
        qiskit.circuit.library.CZGate(),
        qiskit.circuit.library.CHGate(),
        qiskit.circuit.library.CSGate(),
        qiskit.circuit.library.CSdgGate(),
        qiskit.circuit.library.CSXGate(),
        qiskit.circuit.library.CCZGate(),
        qiskit.circuit.library.CPhaseGate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.CRZGate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.CRXGate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.CRYGate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.CU1Gate(rng.uniform(-2 * np.pi, 2 * np.pi)),
        qiskit.circuit.library.CU3Gate(*rng.uniform(-2 * np.pi, 2 * np.pi, size=3)),
        qiskit.circuit.library.CUGate(*rng.uniform(-2 * np.pi, 2 * np.pi, size=4)),
        qiskit.circuit.library.CCXGate(),
        qiskit.circuit.library.MCXGate(3),
        qiskit.circuit.library.MCU1Gate(rng.uniform(-2 * np.pi, 2 * np.pi), 2),
        qiskit.circuit.library.MCPhaseGate(rng.uniform(-2 * np.pi, 2 * np.pi), 2),
        qss.custom_gates.iCCXGate(),
        qss.custom_gates.iCCXdgGate(),
        qss.AQTiCCXGate(),
    ]

    # meta: make sure we've covered every gate type in _known_qiskit_controlled_gate_types
    covered = {gate.base_class for gate in qiskit_gates}
    missing = set(sm.converters.qiskit._known_qiskit_controlled_gate_types).difference(covered)
    assert not missing, "Missing tests for: " + ", ".join(gate.__name__ for gate in missing)

    return qiskit_gates


def trial_keep_definition_gates() -> list[qiskit.circuit.ControlledGate]:
    # An example of every gate in _keep_qiskit_definition
    qiskit_gates = [
        qiskit.circuit.library.C3SXGate(),
        qiskit.circuit.library.C3XGate(),
        qiskit.circuit.library.C4XGate(),
        qiskit.circuit.library.RCCXGate(),
        qiskit.circuit.library.CSwapGate(),
        qiskit.circuit.library.MCXGrayCode(5),
        qiskit.circuit.library.MCXRecursive(3),
        qiskit.circuit.library.MCXVChain(3, 2),
        qiskit.circuit.library.MCXVChain(3),
        qiskit.circuit.library.MCMTGate(qiskit.circuit.library.RYGate(1.23), 2, 2),
    ]

    # meta: make sure we've covered every gate type in _keep_qiskit_definition
    covered = {gate.base_class for gate in qiskit_gates}
    missing = set(sm.converters.qiskit._keep_qiskit_definition).difference(covered)
    assert not missing, "Missing tests for: " + ", ".join(gate.__name__ for gate in missing)
    return qiskit_gates


def test_gate_maps() -> None:
    """Make sure all the global gate dictionaries in sm.converters.qiskit are valid."""
    standard_args = {"self", "label", "duration", "unit"}

    # static gates should have no arguments aside from (possibly) "label"
    for qiskit_gate_type, cirq_gate in sm.converters.qiskit._qiskit_static_gates.items():
        init_args = set(inspect.signature(qiskit_gate_type.__init__).parameters)
        assert init_args.issubset(standard_args)
        if cirq.has_unitary(cirq_gate):
            assert issubclass(qiskit_gate_type, qiskit.circuit.Gate)
            assert _gates_are_equivalent(qiskit_gate_type(), cirq_gate)
        else:
            assert issubclass(qiskit_gate_type, qiskit.circuit.Instruction)

    # pow gates should have a single argument aside from (possibly) "label"
    for qiskit_gate_type, cirq_gate in sm.converters.qiskit._qiskit_pow_gates.items():
        assert issubclass(qiskit_gate_type, qiskit.circuit.Gate)

        init_args = set(inspect.signature(qiskit_gate_type.__init__).parameters)
        assert len(init_args.difference(standard_args)) == 1
        assert _gates_are_equivalent(qiskit_gate_type(np.pi), cirq_gate)

    # make sure there are no duplicates
    all_gate_types = [
        *sm.converters.qiskit._qiskit_static_gates,
        *sm.converters.qiskit._qiskit_pow_gates,
        *sm.converters.qiskit._qiskit_param_gates,
        *sm.converters.qiskit._known_qiskit_controlled_gate_types,
        *sm.converters.qiskit._keep_qiskit_definition,
    ]
    assert len(set(all_gate_types)) == len(all_gate_types)

    for module in (
        qiskit.circuit.controlflow,
        qiskit.circuit.library,
        qss.custom_gates,
    ):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, qiskit.circuit.ControlledGate):
                assert attr in (
                    *sm.converters.qiskit._known_qiskit_controlled_gate_types,
                    *sm.converters.qiskit._keep_qiskit_definition,
                    qiskit.circuit.library.CUGate,
                ), (
                    f"{attr_name} should be in either _known_qiskit_controlled_gate_types or"
                    "_keep_qiskit_definition"
                )


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
def test_swap_endianness(num_qubits: int) -> None:
    matrix = cirq.testing.random_unitary(2**num_qubits)
    swapped_matrix = sm.converters.qiskit.swap_endianness(matrix)

    # Swapped matrix should be the same as a MatrixGate applied to qubits in reverse order
    qs = cirq.LineQubit.range(num_qubits)
    np.testing.assert_array_equal(
        cirq.Circuit(cirq.MatrixGate(matrix).on(*qs[::-1])).unitary(qs), swapped_matrix
    )

    # Applying twice should return the original matrix
    np.testing.assert_array_equal(sm.converters.qiskit.swap_endianness(swapped_matrix), matrix)


@parametrize_over_qiskit_gates(*trial_qiskit_static_gates())
def test_convert_gate_static(trial_gate: qiskit.circuit.Instruction) -> None:
    cirq_gate = sm.converters.qiskit.qiskit_gate_to_cirq_gate(trial_gate)
    assert cirq_gate is sm.converters.qiskit._qiskit_static_gates[trial_gate.base_class]


@parametrize_over_qiskit_gates(*trial_qiskit_pow_gates())
def test_convert_gate_pow(trial_gate: qiskit.circuit.Instruction) -> None:
    cirq_gate = sm.converters.qiskit.qiskit_gate_to_cirq_gate(trial_gate)
    exponent = trial_gate.params[0] / np.pi
    assert cirq_gate == sm.converters.qiskit._qiskit_pow_gates[trial_gate.base_class] ** exponent
    assert _gates_are_equivalent(trial_gate, cirq_gate)


@parametrize_over_qiskit_gates(*trial_qiskit_param_gates())
def test_convert_gate_param(trial_gate: qiskit.circuit.Instruction) -> None:
    cirq_gate = sm.converters.qiskit.qiskit_gate_to_cirq_gate(trial_gate)
    assert cirq_gate == sm.converters.qiskit._qiskit_param_gates[trial_gate.base_class](trial_gate)
    assert _gates_are_equivalent(trial_gate, cirq_gate)


def test_convert_gate_unknown() -> None:
    # qiskit gates with no single-gate cirq equivalence
    for qiskit_gate in (qiskit.circuit.library.QFT(3), qiskit.circuit.library.MCXGrayCode(4)):
        assert sm.converters.qiskit.qiskit_gate_to_cirq_gate(qiskit_gate) is None


@parametrize_over_qiskit_gates(
    *trial_qiskit_static_gates(),
    *trial_qiskit_pow_gates(),
    *trial_qiskit_param_gates(),
    *trial_controlled_gates(),
)
def test_handle_qiskit_instruction_known(trial_gate: qiskit.circuit.Instruction) -> None:
    qubits = random_qubits(trial_gate.num_qubits)
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert phase == 0
    assert isinstance(cirq_op, cirq.Operation)
    if cirq.has_unitary(cirq_op):
        assert _operations_are_equivalent(trial_gate, cirq_op, qubits)

    assert (cirq_op, phase) == sm.converters.qiskit._handle_qiskit_inst(
        trial_gate.to_mutable(), qubits, []
    )


@parametrize_over_qiskit_gates(*trial_keep_definition_gates())
def test_handle_qiskit_instruction_using_qiskit_definition(
    trial_gate: qiskit.circuit.Instruction,
) -> None:
    qubits = random_qubits(trial_gate.num_qubits)
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert isinstance(cirq_op, cirq.CircuitOperation)
    assert (cirq_op, phase) == sm.converters.qiskit._handle_qiskit_definition(
        trial_gate, qubits, []
    )
    assert _operations_are_equivalent(trial_gate, cirq_op, qubits, phase)


def test_handle_qiskit_instruction_parallel_gates() -> None:
    rng = np.random.default_rng()
    qubits = random_qubits(3)

    theta, phi = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
    trial_gate = qiskit.circuit.library.GR(3, theta, phi)[0].operation
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert cirq.approx_eq(cirq_op, css.ParallelRGate(theta, phi, 3).on(*qubits))
    assert phase == 0

    trial_gate = qss.ParallelGates(qiskit.circuit.library.HGate(), qss.AceCR("-+"))
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert cirq_op == css.ParallelGates(cirq.H, css.AceCR("-+")).on(*qubits)
    assert phase == 0


def test_handle_qiskit_instruction() -> None:
    qubits = random_qubits(3)

    # Measurement
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(
        qiskit.circuit.library.Measure(), [cirq.LineQubit(123)], ["abc"]
    )
    assert cirq_op == cirq.measure(cirq.LineQubit(123), key="abc")
    assert phase == 0

    # Operation subclassing QuantumCircuit instead of Instruction
    trial_gate = qiskit.circuit.library.QFT(3).to_instruction()
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert isinstance(cirq_op, cirq.CircuitOperation)
    assert _operations_are_equivalent(trial_gate, cirq_op, qubits, phase)

    # MatrixGate fallback:
    matrix = cirq.testing.random_unitary(8)
    trial_gate = qiskit.circuit.library.UnitaryGate(matrix)
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, [])
    assert isinstance(cirq_op.gate, cirq.MatrixGate)
    assert _operations_are_equivalent(trial_gate, cirq_op, qubits)
    assert phase == 0

    with pytest.raises(gss.SuperstaqException, match="We don't know what to do"):
        _, _ = sm.converters.qiskit._handle_qiskit_inst(
            qiskit.circuit.Gate("unknown", 1, []), [cirq.LineQubit(789)], []
        )


def test_translate_circuit() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)

    qiskit_circuit = qiskit.QuantumCircuit(4, 4)
    qiskit_circuit.cx(1, 2)
    qiskit_circuit.cx(1, 3, label="no_compile")
    qiskit_circuit.measure(1, 1)
    qiskit_circuit.measure(2, 0)
    qiskit_circuit.measure(3, 3)

    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit)
    cirq.testing.assert_same_circuits(
        cirq_circuit,
        cirq.Circuit(
            cirq.CX(q1, q2),
            cirq.CX(q1, q3).with_tags("no_compile"),
            cirq.measure(q1, key="c_1"),
            cirq.measure(q2, key="c_0"),
            cirq.measure(q3, key="c_3"),
        ),
    )

    qiskit_circuit = qiskit.QuantumCircuit(
        qiskit.QuantumRegister(name="x", size=2),
        qiskit.QuantumRegister(name="y", size=2),
        qiskit.ClassicalRegister(name="a", size=2),
        qiskit.ClassicalRegister(name="b", size=2),
    )
    qiskit_circuit.measure(1, 1)
    qiskit_circuit.measure(2, 0)
    qiskit_circuit.measure(3, 3)
    qiskit_circuit.global_phase = 1.23

    expected = cirq.Circuit(
        cirq.measure(q1, key="a_1"),
        cirq.measure(q2, key="a_0"),
        cirq.measure(q3, key="b_1"),
    )

    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit)
    cirq.testing.assert_same_circuits(cirq_circuit, expected)

    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit, keep_global_phase=True)
    cirq.testing.assert_same_circuits(
        cirq_circuit, (expected + cirq.global_phase_operation(np.exp(1j * 1.23)))
    )

    # Make sure sure classical registers named "measure" will convert correctly in spite of
    # https://github.com/quantumlib/Cirq/issues/5262
    qiskit_circuit = qiskit.QuantumCircuit(
        qiskit.QuantumRegister(4),
        qiskit.ClassicalRegister(1, "somename"),
        qiskit.ClassicalRegister(2, "measure"),
        qiskit.ClassicalRegister(1, "meas"),
    )
    qiskit_circuit.measure(1, 2)
    qiskit_circuit.measure(2, 1)
    qiskit_circuit.measure(3, 0)
    qiskit_circuit.measure(0, 3)

    expected_cirq_circuit = cirq.Circuit(
        cirq.measure(q0, key="meas_0"),
        cirq.measure(q1, key="measure_1"),
        cirq.measure(q2, key="measure_0"),
        cirq.measure(q3, key="somename_0"),
    )
    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit)
    cirq.testing.assert_same_circuits(cirq_circuit, expected_cirq_circuit)

    qiskit_circuit = qiskit.QuantumCircuit(4)
    qiskit_circuit.delay(1.2, 0, unit="ps")
    qiskit_circuit.delay(3.4, [1, 2], unit="ms")
    qiskit_circuit.append(qiskit.circuit.Delay(5.6, unit="s"), [3])

    expected_cirq_circuit = cirq.Circuit(
        cirq.WaitGate(cirq.Duration(picos=1.2)).on(q0),
        cirq.WaitGate(cirq.Duration(millis=3.4)).on(q1),
        cirq.WaitGate(cirq.Duration(millis=3.4)).on(q2),
        cirq.WaitGate(cirq.Duration(millis=5.6e3)).on(q3),
    )
    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit)
    cirq.testing.assert_same_circuits(cirq_circuit, expected_cirq_circuit)


def test_translate_circuit_with_labels() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)

    qiskit_circuit = qiskit.QuantumCircuit(4)
    qiskit_circuit.x(0)
    qiskit_circuit.x(1, label="foo")
    qiskit_circuit.x([2, 3], label="bar")
    qiskit_circuit.dcx(0, 1)
    qiskit_circuit.append(qiskit.circuit.library.DCXGate(label="no_compile"), [1, 2])
    qiskit_circuit.append(qiskit.circuit.library.DCXGate(label="pls_compile"), [2, 3])

    cirq_circuit = sm.converters.qiskit_to_cirq(qiskit_circuit)
    cirq.testing.assert_same_circuits(
        cirq_circuit,
        cirq.Circuit(
            cirq.X(q0),
            cirq.X(q1).with_tags("foo"),
            cirq.X(q2).with_tags("bar"),
            cirq.X(q3).with_tags("bar"),
            cirq.CX(q0, q1),
            cirq.CX(q1, q0),
            cirq.CircuitOperation(
                cirq.Circuit(cirq.CX(q1, q2), cirq.CX(q2, q1)).freeze()
            ).with_tags("no_compile"),
            cirq.CX(q2, q3).with_tags("pls_compile"),
            cirq.CX(q3, q2).with_tags("pls_compile"),
        ),
    )


def test_classical_bit_padding() -> None:
    qiskit_circuit = qiskit.QuantumCircuit(1, 1)
    qiskit_circuit.measure(0, 0)
    cirq.testing.assert_same_circuits(
        sm.converters.qiskit_to_cirq(qiskit_circuit),
        cirq.Circuit(cirq.measure(cirq.q(0), key="c_0")),
    )

    qiskit_circuit = qiskit.QuantumCircuit(1, 10)
    qiskit_circuit.measure(0, 0)
    qiskit_circuit.measure(0, 9)
    cirq.testing.assert_same_circuits(
        sm.converters.qiskit_to_cirq(qiskit_circuit),
        cirq.Circuit(cirq.measure(cirq.q(0), key="c_0"), cirq.measure(cirq.q(0), key="c_9")),
    )

    qiskit_circuit = qiskit.QuantumCircuit(1, 11)
    qiskit_circuit.measure(0, 0)
    qiskit_circuit.measure(0, 9)
    cirq.testing.assert_same_circuits(
        sm.converters.qiskit_to_cirq(qiskit_circuit),
        cirq.Circuit(cirq.measure(cirq.q(0), key="c_00"), cirq.measure(cirq.q(0), key="c_09")),
    )

    qiskit_circuit = qiskit.QuantumCircuit(1, 101)
    qiskit_circuit.measure(0, 0)
    qiskit_circuit.measure(0, 90)
    cirq.testing.assert_same_circuits(
        sm.converters.qiskit_to_cirq(qiskit_circuit),
        cirq.Circuit(cirq.measure(cirq.q(0), key="c_000"), cirq.measure(cirq.q(0), key="c_090")),
    )

    # The following fails without padding due to bit reordering
    qiskit_circuit = qiskit.QuantumCircuit(
        qiskit.QuantumRegister(20, "q"),
        qiskit.ClassicalRegister(20, "c"),
    )
    qiskit_circuit.measure(range(20), range(20))
    round_trip_circuit = sm.converters.cirq_to_qiskit(
        sm.converters.qiskit_to_cirq(qiskit_circuit), cirq.LineQubit.range(20)
    )
    assert round_trip_circuit == qiskit_circuit


def test_handle_qiskit_circuit() -> None:
    rng = np.random.default_rng()

    trial_gates = [
        *trial_controlled_gates(),
        *trial_qiskit_pow_gates(),
        *trial_qiskit_param_gates(),
        qiskit.circuit.library.QFT(3),
        qiskit.circuit.library.GR(3, *rng.uniform(-2 * np.pi, 2 * np.pi, size=2)),
        qss.ParallelGates(qiskit.circuit.library.HGate(), qss.AceCR("-+")),
    ]

    for qiskit_gate in trial_qiskit_static_gates():
        # this test compares the unitary effect of each circuit, so skip over nonunitary ops
        if cirq.has_unitary(sm.converters.qiskit._qiskit_static_gates[qiskit_gate.base_class]):
            trial_gates.append(qiskit_gate)

    total_qubits = max(qiskit_gate.num_qubits for qiskit_gate in trial_gates)

    # make a random circuit
    qiskit_circuit = qiskit.QuantumCircuit(total_qubits)
    qiskit_circuit.global_phase = 1.23
    for trial_gate_index in rng.permutation(len(trial_gates)):
        qiskit_gate = trial_gates[trial_gate_index]
        qubits = rng.choice(qiskit_circuit.num_qubits, size=qiskit_gate.num_qubits, replace=False)
        qiskit_circuit.append(qiskit_gate, qubits.tolist())

    cirq_qubits = random_qubits(qiskit_circuit.num_qubits)
    cirq_circuit, phase = sm.converters.qiskit._handle_qiskit_circuit(
        qiskit_circuit, cirq_qubits, []
    )
    assert _operations_are_equivalent(qiskit_circuit, cirq_circuit, cirq_qubits, phase)

    # circuit w/ recursion
    qiskit_circuit.append(
        qiskit_circuit.to_instruction(), rng.permutation(qiskit_circuit.num_qubits).tolist()
    )
    qiskit_circuit.append(
        qiskit_circuit.to_instruction(), rng.permutation(qiskit_circuit.num_qubits).tolist()
    )
    cirq_circuit, phase2 = sm.converters.qiskit._handle_qiskit_circuit(
        qiskit_circuit, cirq_qubits, []
    )
    assert _operations_are_equivalent(qiskit_circuit, cirq_circuit, cirq_qubits, phase2)
    assert np.isclose(phase2, 4 * phase)


def test_conditional_sub_circuit() -> None:
    qiskit_circuit = qiskit.QuantumCircuit(4, 1)
    qiskit_circuit.h([0, 1, 2, 3])
    qiskit_circuit.cx(0, 1)
    qiskit_circuit.measure(0, 0)
    with qiskit_circuit.if_test((qiskit_circuit.clbits[0], 0b1)):
        # Conditional sub-circuit (more than one operation)
        qiskit_circuit.x(1)
        qiskit_circuit.y(2)
        qiskit_circuit.z(3)
        qiskit_circuit.ccz(1, 3, 2)
    qiskit_circuit.t(1)

    qubits = [cirq.GridQubit(1, 4), cirq.LineQubit(1), cirq.LineQubit(3), cirq.NamedQubit("jay")]
    cirq_circuit, _ = sm.converters.qiskit._handle_qiskit_circuit(qiskit_circuit, qubits, ["meas"])
    expected_circuit = textwrap.dedent(
        """
                               ┌───┐
        (1, 4): ───H───@───M───────────────────
                       │   ║
        1: ────────H───X───╫────X──────@───T───
                           ║    ║      ║
        3: ────────H───────╫────╫Y─────@───────
                           ║    ║║     ║
        jay: ──────H───────╫────╫╫Z────@───────
                           ║    ║║║    ║
        meas: ═════════════@════^^^════^═══════
                               └───┘
        """
    )
    cirq.testing.assert_has_diagram(cirq_circuit, expected_circuit)


def test_handle_qiskit_circuit_with_classical_bits() -> None:
    # teleportation circuit
    qiskit_circuit = qiskit.QuantumCircuit(4, 4)
    qiskit_circuit.h(2)
    qiskit_circuit.cx(2, 3)
    qiskit_circuit.cx(0, 2)
    qiskit_circuit.barrier(0, 2, 3)
    qiskit_circuit.reset([0, 2, 3])
    qiskit_circuit.h(0)
    qiskit_circuit.measure([0, 2], [1, 2])
    with qiskit_circuit.if_test((qiskit_circuit.clbits[1], 0b1)):
        qiskit_circuit.z(3)
    with qiskit_circuit.if_test((qiskit_circuit.clbits[2], 0b1)):
        qiskit_circuit.x(3)

    qubits = [cirq.GridQubit(1, 4), cirq.LineQubit(1), cirq.LineQubit(3), cirq.NamedQubit("jay")]
    measurement_keys = ["A", "B", "C", "D"]

    cirq_circuit, _ = sm.converters.qiskit._handle_qiskit_circuit(
        qiskit_circuit, qubits, measurement_keys
    )

    expected_circuit = textwrap.dedent(
        """
        (1, 4): ───────────@───│───R───H───M───────────
                           │   │           ║
        3: ────────H───@───X───│───R───M───╫───────────
                       │       │       ║   ║
        jay: ──────────X───────│───R───╫───╫───Z───X───
                                       ║   ║   ║   ║
        B: ════════════════════════════╬═══@═══^═══╬═══
                                       ║           ║
        C: ════════════════════════════@═══════════^═══
        """
    )
    cirq.testing.assert_has_diagram(cirq_circuit, expected_circuit)

    # do the same thing but w/ subcircuits to make sure qubits/clbits are mapped correctly
    x_flip_circuit = qiskit.QuantumCircuit(1, 1)
    with x_flip_circuit.if_test((x_flip_circuit.cregs[0], 1)):
        x_flip_circuit.x(0)

    x_adj_circuit = qiskit.QuantumCircuit(2, 1)
    x_adj_circuit.measure(0, 0)
    x_adj_circuit = x_adj_circuit.compose(x_flip_circuit, [1], [0])

    z_flip_circuit = qiskit.QuantumCircuit(1, 1)
    with z_flip_circuit.if_test((z_flip_circuit.cregs[0], 1)):
        z_flip_circuit.z(0)

    z_adj_circuit = qiskit.QuantumCircuit(2, 1)
    z_adj_circuit.h(0)
    z_adj_circuit.measure(0, 0)
    z_adj_circuit = z_adj_circuit.compose(z_flip_circuit, [1], [0])

    qiskit_circuit = qiskit.QuantumCircuit(4, 4)
    qiskit_circuit.h(2)
    qiskit_circuit.cx(2, 3)
    qiskit_circuit.cx(0, 2)
    qiskit_circuit.barrier(0, 2, 3)
    qiskit_circuit.reset([0, 2, 3])
    qiskit_circuit = qiskit_circuit.compose(z_adj_circuit, [0, 3], [1])
    qiskit_circuit = qiskit_circuit.compose(x_adj_circuit, [2, 3], [2])

    cirq_circuit, _ = sm.converters.qiskit._handle_qiskit_circuit(
        qiskit_circuit, qubits, measurement_keys
    )
    cirq.testing.assert_has_diagram(cirq_circuit, expected_circuit)

    # Other control values
    qiskit_circuit = qiskit.QuantumCircuit(2, 2)
    qiskit_circuit.measure(1, 0)
    qiskit_circuit.measure(0, 1)

    # controlled by single Clbits
    with qiskit_circuit.if_test((qiskit_circuit.clbits[0], 0b1)):
        qiskit_circuit.z(1)
    with qiskit_circuit.if_test((qiskit_circuit.clbits[1], 0b0)):
        qiskit_circuit.x(0)

    # controlled by full register
    with qiskit_circuit.if_test((qiskit_circuit.cregs[0], 0b11)):  # clbit[0] == clbit[1] == 1
        qiskit_circuit.x(1)
    with qiskit_circuit.if_test((qiskit_circuit.cregs[0], 0b10)):  # clbit[0] == 0, clbit[1] == 1
        qiskit_circuit.z(0)

    qubits = [cirq.LineQubit(5), cirq.LineQubit(2)]
    cirq_circuit, _ = sm.converters.qiskit._handle_qiskit_circuit(
        qiskit_circuit, qubits, ["B", "A"]
    )
    cirq.testing.assert_has_diagram(
        cirq_circuit,
        textwrap.dedent(
            """
                  ┌──┐   ┌──────────────────────┐   ┌─────────────────────────┐
            2: ────M──────Z──────────────────────────X────────────────────────────
                   ║      ║                          ║
            5: ────╫M─────╫X(conditions=[1 - A])─────╫Z(conditions=[1 - B, A])────
                   ║║     ║║                         ║║
            A: ════╬@═════╬^═════════════════════════^^═══════════════════════════
                   ║      ║                          ║║
            B: ════@══════^══════════════════════════^^═══════════════════════════
                  └──┘   └──────────────────────┘   └─────────────────────────┘
            """
        ),
    )


@parametrize_over_qiskit_gates(*trial_controlled_gates())
def test_handle_qiskit_controlled_op(trial_gate: qiskit.circuit.ControlledGate) -> None:
    while trial_gate.ctrl_state > 0:
        qubits = random_qubits(trial_gate.num_qubits)
        cirq_op = sm.converters.qiskit._handle_qiskit_controlled_op(trial_gate, qubits)
        assert sm.converters.qiskit._handle_qiskit_inst(trial_gate, qubits, []) == (cirq_op, 0.0)
        assert isinstance(cirq_op, cirq.Operation)
        assert _operations_are_equivalent(trial_gate, cirq_op, qubits)

        trial_gate = trial_gate.to_mutable()
        trial_gate.ctrl_state >>= 1


@pytest.mark.parametrize(
    "base_gate",
    [
        qiskit.circuit.library.ZGate(),
        qiskit.circuit.library.SGate(),
        qiskit.circuit.library.SdgGate(),
        qiskit.circuit.library.TGate(),
        qiskit.circuit.library.TdgGate(),
        qiskit.circuit.library.PhaseGate(1.23),
        qiskit.circuit.library.U1Gate(1.23),
        qiskit.circuit.library.RZGate(1.23),
        qiskit.circuit.library.XGate(),
        qiskit.circuit.library.SXGate(),
        qiskit.circuit.library.SXdgGate(),
        qiskit.circuit.library.RXGate(1.23),
    ],
)
def test_handle_qiskit_controlled_op_uses_correct_sub_op(base_gate: qiskit.circuit.Gate) -> None:
    for num_controls in (2, 3, 4):
        ctrl_state = 1 << (num_controls - 1)
        qiskit_gate = base_gate.control(num_controls, ctrl_state=ctrl_state)

        qubits = cirq.LineQubit.range(qiskit_gate.num_qubits)
        cirq_op = sm.converters.qiskit._handle_qiskit_controlled_op(qiskit_gate, qubits)
        expected_base_gate = sm.converters.qiskit.qiskit_gate_to_cirq_gate(base_gate)
        assert isinstance(cirq_op, cirq.ControlledOperation)
        assert cirq_op.sub_operation.gate == expected_base_gate
        assert _operations_are_equivalent(qiskit_gate, cirq_op, qubits)


def test_handle_qiskit_controlled_op_with_unknown_gates() -> None:
    q0, q1, q2, q3 = qubits = random_qubits(4)
    qiskit_gate = qss.AceCR("+-").control(2, ctrl_state="10")
    expected = css.AceCR("+-").on(q2, q3).controlled_by(q0, q1, control_values=[0, 1])
    assert sm.converters.qiskit._handle_qiskit_controlled_op(qiskit_gate, qubits) == expected
    assert sm.converters.qiskit._handle_qiskit_inst(qiskit_gate, qubits, []) == (expected, 0.0)

    qiskit_gate = qiskit.circuit.library.GR(2, 1.1, 2.2).to_gate().control(2, ctrl_state="11")
    expected = css.ParallelRGate(1.1, 2.2, 2).on(q2, q3).controlled_by(q0, q1)
    assert sm.converters.qiskit._handle_qiskit_controlled_op(qiskit_gate, qubits) == expected
    assert sm.converters.qiskit._handle_qiskit_inst(qiskit_gate, qubits, []) == (expected, 0.0)

    # control a whole circuit
    circuit = qiskit.QuantumCircuit(3)
    circuit.x(0)
    circuit.append(qss.ZZSwapGate(1.2), [0, 1])
    circuit.cz(1, 2)
    circuit.cx(1, 2)
    qiskit_gate = circuit.to_gate().control(1)

    expected = cirq.CircuitOperation(
        cirq.Circuit(
            cirq.CX(q0, q1),
            css.ZZSwapGate(1.2).on(q1, q2).controlled_by(q0),
            cirq.CCZ(q0, q2, q3),
            cirq.CCX(q0, q2, q3),
        ).freeze()
    )
    assert sm.converters.qiskit._handle_qiskit_controlled_op(qiskit_gate, qubits) == expected
    assert sm.converters.qiskit._handle_qiskit_inst(qiskit_gate, qubits, []) == (expected, 0.0)

    qiskit_gate.definition = qiskit.circuit.library.C3XGate().definition
    assert sm.converters.qiskit._handle_qiskit_controlled_op(qiskit_gate, qubits) is None


def test_handle_qiskit_u_gate() -> None:
    rng = np.random.default_rng()

    theta, phi, lam = rng.uniform(-2 * np.pi, 2 * np.pi, size=3)
    qiskit_gate = qiskit.circuit.library.UGate(theta, phi, lam)
    cirq_gate, phase = sm.converters.qiskit._handle_qiskit_u_gate(theta, phi, lam)
    assert cirq.allclose_up_to_global_phase(qiskit_gate.to_matrix(), cirq.unitary(cirq_gate))
    assert np.allclose(qiskit_gate.to_matrix(), cirq.unitary(cirq_gate) * np.exp(1j * phase))

    qiskit_gate = qiskit.circuit.library.UGate(0.0, phi, lam)
    cirq_gate, phase = sm.converters.qiskit._handle_qiskit_u_gate(0.0, phi, lam)
    assert isinstance(cirq_gate, cirq.ZPowGate)
    assert not phase
    assert np.allclose(qiskit_gate.to_matrix(), cirq.unitary(cirq_gate))

    qiskit_gate = qiskit.circuit.library.UGate(theta, phi, -phi)
    cirq_gate, phase = sm.converters.qiskit._handle_qiskit_u_gate(theta, phi, -phi)
    assert isinstance(cirq_gate, cirq.PhasedXPowGate)
    assert not phase
    assert np.allclose(qiskit_gate.to_matrix(), cirq.unitary(cirq_gate))

    qiskit_gate = qiskit.circuit.library.UGate(np.pi, phi, lam)
    cirq_gate, phase = sm.converters.qiskit._handle_qiskit_u_gate(np.pi, phi, lam)
    assert isinstance(cirq_gate, cirq.PhasedXZGate)
    assert not phase
    assert np.allclose(qiskit_gate.to_matrix(), cirq.unitary(cirq_gate))


def test_get_cirq_duration() -> None:
    assert sm.converters.qiskit._get_cirq_duration(1.1, "ps") == cirq.Duration(picos=1.1)
    assert sm.converters.qiskit._get_cirq_duration(2.1, "ns") == cirq.Duration(nanos=2.1)
    assert sm.converters.qiskit._get_cirq_duration(3.1, "us") == cirq.Duration(micros=3.1)
    assert sm.converters.qiskit._get_cirq_duration(4.1, "ms") == cirq.Duration(millis=4.1)
    assert sm.converters.qiskit._get_cirq_duration(5.1, "s") == cirq.Duration(millis=5100)

    with pytest.raises(gss.SuperstaqException, match=r"We only.*\(got 'dt'\)"):
        _ = sm.converters.qiskit._get_cirq_duration(123, "dt")


def test_handle_qiskit_definition() -> None:
    q0 = cirq.LineQubit(5)
    q1 = cirq.LineQubit(2)

    qiskit_gate = qss.ZZSwapGate(1.23)
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_definition(qiskit_gate, [q0, q1], [])

    assert phase == 0.0
    assert isinstance(cirq_op, cirq.CircuitOperation)
    cirq.testing.assert_same_circuits(
        cirq_op.mapped_circuit(), cirq.Circuit(cirq.decompose_once(css.ZZSwapGate(1.23).on(q0, q1)))
    )

    qiskit_circuit = qiskit.QuantumCircuit(5, 5)
    qiskit_circuit.rx(np.pi / 2, 1)
    qiskit_circuit.h(3)
    qiskit_circuit.cx(3, 1)
    qiskit_circuit.measure(3, 3)
    qiskit_circuit.measure(1, 4)
    qiskit_circuit.global_phase = 1.25

    qubits = random_qubits(5)
    measurement_keys = ["A", "B", "C", "D", "E"]
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_definition(
        qiskit_circuit.to_instruction(), qubits, measurement_keys
    )

    expected_cirq_circuit = cirq.Circuit(
        cirq.rx(np.pi / 2).on(qubits[1]),
        cirq.H(qubits[3]),
        cirq.CX(qubits[3], qubits[1]),
        cirq.measure(qubits[3], key="D"),
        cirq.measure(qubits[1], key="E"),
    )

    assert phase == 1.25
    assert isinstance(cirq_op, cirq.CircuitOperation)
    cirq.testing.assert_same_circuits(cirq_op.mapped_circuit(), expected_cirq_circuit)

    qiskit_gate = qss.ParallelGates(qss.ZZSwapGate(1.23), qiskit.circuit.library.XGate())
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_definition(qiskit_gate, qubits[:3], [])

    assert phase == 0.0
    assert cirq_op == css.ParallelGates(css.ZZSwapGate(1.23), cirq.X).on(*qubits[:3])

    qiskit_gate = qiskit.circuit.library.GR(3, 1.23, 4.56)[0].operation
    cirq_op, phase = sm.converters.qiskit._handle_qiskit_definition(qiskit_gate, qubits[-3:], [])

    assert phase == 0.0
    assert cirq_op == css.ParallelRGate(1.23, 4.56, 3).on(*qubits[-3:])


def test_intercept_qiskit_parallel_r_gate() -> None:
    q0, q1, q2, q3 = random_qubits(4)
    cirq_circuit = cirq.Circuit(css.RGate(0.2, 0.3).on_each(q0, q1, q2))

    cirq_op = sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq_circuit)
    assert cirq_op == css.ParallelRGate(0.2, 0.3, 3).on(q0, q1, q2)

    cirq_circuit1 = cirq_circuit + css.RGate(0.2, 0.3).on(q3)
    cirq_op = sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq_circuit1)
    assert cirq_op == css.ParallelRGate(0.2, 0.3, 4).on(q0, q1, q2, q3)

    cirq_circuit1 = cirq_circuit + css.RGate(0.2, 0.4).on(q3)  # RGate with different parameters
    assert sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq_circuit1) is None

    cirq_circuit1 = cirq_circuit + cirq.X(q3)  # non-RGate on q3
    assert sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq_circuit1) is None

    cirq_circuit1 = cirq_circuit + css.RGate(0.2, 0.3).on(q0)  # two RGates on q0
    assert sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq_circuit1) is None

    assert sm.converters.qiskit._intercept_qiskit_parallel_r_gate(cirq.Circuit()) is None


def test_parameterized_gates() -> None:
    param_angle = qiskit.circuit.Parameter("theta")
    # As of `qiskit>=2.1`, a `qiskit.circuit.ParameterExprsmion` should not be directly
    # instantiated. Instead, it can be created indirectly via `qiskit.circuit.Parameter`; for
    # example:
    param_expr = 1 + 5 * param_angle
    assert isinstance(param_expr, qiskit.circuit.ParameterExpression)

    test_params = (
        1.23,
        param_expr.bind({param_angle: 0.046}),
        param_expr.bind({param_angle: 0.046 + (10**-10) * 1j}),
    )
    for param in test_params:
        param_gate = qiskit.circuit.library.RXGate(np.pi * param)
        sm.converters.qiskit._parameter_expressions_to_float(param_gate)
        assert param_gate == qiskit.circuit.library.RXGate(1.23 * np.pi)


def test_parameterized_circuits() -> None:
    q0 = cirq.LineQubit(0)
    expected_cirq_circuit = cirq.Circuit(
        cirq.PhasedXZGate(
            axis_phase_exponent=0.405,
            x_exponent=0.0318,
            z_exponent=0.159,
        ).on(q0),
        cirq.Rx(rads=1.0).on(q0),
        cirq.Rz(rads=0.1).on(q0),
        css.RGate(0.1, 0.3).on(q0),
    )
    theta = qiskit.circuit.Parameter("θ")
    beta = qiskit.circuit.Parameter("β")
    phi = qiskit.circuit.Parameter("Φ")

    qc = qiskit.QuantumCircuit(1, 1)
    qc.u(theta, beta, phi, 0)

    with pytest.raises(
        gss.SuperstaqException,
        match="Can't convert parameterized unbounded qiskit circuits",
    ):
        _ = sm.converters.qiskit_to_cirq(qc)

    qc.rx(1 + 0j * theta, 0)
    qc.rz(theta, 0)
    qc.r(theta, phi, 0)
    binded_circuit = qc.assign_parameters({theta: 0.1, beta: 0.2, phi: 0.3})
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        sm.converters.qiskit_to_cirq(binded_circuit),
        expected_cirq_circuit,
        1e-3,
    )

    with pytest.raises(
        gss.SuperstaqException,
        match="Can't convert parameterized unbounded qiskit circuits",
    ):
        qc.rx(1 + 1j * theta, 0)
        _ = sm.converters.qiskit_to_cirq(qc)


def test_qiskit_quantum_volume_circuit() -> None:
    """Make sure https://github.com/SupertechLabs/qiskit-superstaq/issues/130 has been fixed."""
    # Same circuit as qiskit_experiments.quantum_volume.QuantumVolume([0, 1, 2, 3], trials=1)[0]:
    circuit = qiskit.circuit.library.QuantumVolume(4)
    circuit.measure_active()

    _ = sm.converters.qiskit_to_cirq(circuit)

    circuit = qiskit.transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"], optimization_level=0)
    _ = sm.converters.qiskit_to_cirq(circuit)


@pytest.mark.parametrize(
    "cirq_gate",
    sm.converters.qiskit._cirq_static_gates,
    ids=map(str, sm.converters.qiskit._cirq_static_gates),
)
def test_cirq_gate_to_qiskit_gate_static(cirq_gate: cirq.Gate) -> None:
    qiskit_gate = sm.converters.cirq_gate_to_qiskit_gate(cirq_gate)
    assert qiskit_gate == sm.converters.qiskit._cirq_static_gates[cirq_gate]()
    assert qiskit_gate.num_qubits == qiskit_gate.num_qubits
    if cirq.has_unitary(cirq_gate):
        assert _gates_are_equivalent(qiskit_gate, cirq_gate)


@pytest.mark.parametrize(
    "cirq_gate_type",
    sm.converters.qiskit._cirq_pow_gates.keys(),
    ids=[gate_type.__name__ for gate_type in sm.converters.qiskit._cirq_pow_gates],
)
def test_cirq_gate_to_qiskit_gate_pow(cirq_gate_type: type[cirq.EigenGate]) -> None:
    global_shift = 0.0 if cirq_gate_type in [cirq.CZPowGate, css.DDPowGate] else -0.5
    cirq_gate = cirq_gate_type(exponent=0.25, global_shift=global_shift)
    qiskit_gate = sm.converters.cirq_gate_to_qiskit_gate(cirq_gate)
    assert qiskit_gate == sm.converters.qiskit._cirq_pow_gates[cirq_gate_type](np.pi / 4)
    assert _gates_are_equivalent(qiskit_gate, cirq_gate)


@pytest.mark.parametrize(
    ("cirq_gate", "expected_qiskit_gate"),
    [
        (
            cirq.circuits.qasm_output.QasmUGate(0.1, 0.2, 0.3),
            qiskit.circuit.library.UGate(0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi),
        ),
        (css.RGate(0.75, 0.75), qiskit.circuit.library.RGate(0.75, 0.75)),
        (css.ParallelRGate(0.75, 0.75, 2), qiskit.circuit.library.GR(2, 0.75, 0.75)),
        (
            cirq.PhasedXPowGate(exponent=0.2, phase_exponent=-0.25),
            qiskit.circuit.library.RGate(np.pi / 5, -np.pi / 4),
        ),
        (css.Barrier(3), qiskit.circuit.library.Barrier(3)),
        (css.ZZSwapGate(1.23), qss.ZZSwapGate(1.23)),
        (css.CR, qiskit.circuit.library.RZXGate(np.pi)),
        (css.AceCR("+-"), qiskit.circuit.library.ECRGate()),
        (css.AceCR("-+"), qss.AceCR("-+")),
        (css.AceCR("+-", sandwich_rx_rads=1.23), qss.AceCR("+-", sandwich_rx_rads=1.23)),
        (css.AceCR("-+", sandwich_rx_rads=2.31), qss.AceCR("-+", sandwich_rx_rads=2.31)),
        (
            css.ParallelGates(cirq.X, css.AceCR("+-"), css.AceCR("-+"), cirq.rz(1.25 * np.pi)),
            qss.ParallelGates(
                qiskit.circuit.library.XGate(),
                qiskit.circuit.library.ECRGate(),
                qss.AceCR("-+"),
                qiskit.circuit.library.RZGate(1.25 * np.pi),
            ),
        ),
        (
            cirq.MatrixGate(cirq.unitary(cirq.ry(1.2).controlled(1))),
            qiskit.circuit.library.UnitaryGate(qiskit.circuit.library.CRYGate(1.2).to_matrix()),
        ),
        (
            cirq.QubitPermutationGate([1, 2, 0, 3]),
            qiskit.circuit.library.PermutationGate([2, 0, 1, 3]),
        ),
    ],
)
def test_cirq_gate_to_qiskit_gate_param(
    cirq_gate: cirq.Gate, expected_qiskit_gate: qiskit.circuit.Gate
) -> None:
    assert sm.converters.cirq_gate_to_qiskit_gate(cirq_gate) == expected_qiskit_gate
    assert _gates_are_equivalent(expected_qiskit_gate, cirq_gate, ignore_global_phase=True)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
def test_cirq_gate_to_qiskit_gate_matrix(num_qubits: int) -> None:
    matrix = cirq.testing.random_unitary(2**num_qubits)
    cirq_gate = cirq.MatrixGate(matrix, name=f"{num_qubits}-qubit matrix")

    qiskit_gate = sm.converters.cirq_gate_to_qiskit_gate(cirq_gate)
    assert _gates_are_equivalent(qiskit_gate, cirq_gate, ignore_global_phase=False)
    assert isinstance(qiskit_gate, qiskit.circuit.library.UnitaryGate)
    assert qiskit_gate.label == f"{num_qubits}-qubit matrix"


@pytest.mark.parametrize(
    ("cirq_gate", "expected_qiskit_gate"),
    [
        (cirq.X, qiskit.circuit.library.XGate()),
        (cirq.Y, qiskit.circuit.library.YGate()),
        (cirq.Z, qiskit.circuit.library.ZGate()),
        (cirq.X**0.5, qiskit.circuit.library.SXGate()),
        (cirq.S, qiskit.circuit.library.SGate()),
        (cirq.S**-1, qiskit.circuit.library.SdgGate()),
        (cirq.rx(np.pi), qiskit.circuit.library.RXGate(np.pi)),
        (cirq.ry(np.pi), qiskit.circuit.library.RYGate(np.pi)),
        (cirq.rz(np.pi), qiskit.circuit.library.RZGate(np.pi)),
        (cirq.rx(np.pi / 2), qiskit.circuit.library.RXGate(np.pi / 2)),
        (cirq.rz(np.pi / 2), qiskit.circuit.library.RZGate(np.pi / 2)),
        (cirq.rz(-np.pi / 2), qiskit.circuit.library.RZGate(-np.pi / 2)),
    ],
)
def test_cirq_gate_to_qiskit_gate_phase_dependent(
    cirq_gate: cirq.Gate, expected_qiskit_gate: qiskit.circuit.Gate
) -> None:
    """These gates should be mapped differently depending on e.g. whether they're passed as
    cirq.Z or cirq.rz(pi). These will preserve global phase.
    """
    assert sm.converters.cirq_gate_to_qiskit_gate(cirq_gate) == expected_qiskit_gate
    assert _gates_are_equivalent(expected_qiskit_gate, cirq_gate)


@pytest.mark.parametrize(
    ("cirq_gate", "expected_qiskit_gate"),
    [
        (cirq.X**-0.5, qiskit.circuit.library.RXGate(-np.pi / 2)),
        (cirq.T, qiskit.circuit.library.RZGate(np.pi / 4)),
        (cirq.T**-1, qiskit.circuit.library.RZGate(-np.pi / 4)),
        (cirq.XX**0.25, qiskit.circuit.library.RXXGate(np.pi / 4)),
        (cirq.rx(-np.pi / 2), qiskit.circuit.library.RXGate(-np.pi / 2)),
        (cirq.rz(np.pi / 4), qiskit.circuit.library.RZGate(np.pi / 4)),
        (cirq.rz(-np.pi / 4), qiskit.circuit.library.RZGate(-np.pi / 4)),
        (cirq.ms(np.pi / 8), qiskit.circuit.library.RXXGate(np.pi / 4)),
        (
            cirq.X.controlled(4, control_values=[0, 1, 1, 1]),
            qiskit.circuit.library.MCXGate(4, ctrl_state="1110"),
        ),
        (cirq.ISWAP.controlled(2), qiskit.circuit.library.iSwapGate().control(2)),
    ],
)
def test_cirq_gate_to_qiskit_gate_phase_independent(
    cirq_gate: cirq.Gate, expected_qiskit_gate: qiskit.circuit.Gate
) -> None:
    """The gates should not be mapped differently regardless of global phase."""
    assert sm.converters.cirq_gate_to_qiskit_gate(cirq_gate) == expected_qiskit_gate
    assert _gates_are_equivalent(expected_qiskit_gate, cirq_gate, ignore_global_phase=True)


def test_cirq_gate_to_qiskit_gate_unknown() -> None:
    cirq_gate = cirq.FSimGate(1.1, 2.2)
    qiskit_gate = sm.converters.cirq_gate_to_qiskit_gate(cirq_gate)
    assert _gates_are_equivalent(qiskit_gate, cirq_gate, ignore_global_phase=True)

    qubits = cirq.LineQid.for_gate(cirq_gate)
    assert qiskit_gate.definition == sm.converters.cirq_to_qiskit(
        cirq.Circuit(cirq.decompose_once(cirq_gate(*qubits))), qubits
    )

    with pytest.raises(NotImplementedError, match="Unable to convert"):
        _ = sm.converters.cirq_gate_to_qiskit_gate(css.CZ3)


def test_cirq_to_qiskit() -> None:
    q1, q2, q3 = cirq.LineQubit.range(1, 4)
    cirq_circuit = cirq.Circuit(
        cirq.I(q3),
        cirq.X(q3),
        cirq.PhasedXZGate(x_exponent=0.1, z_exponent=0.5, axis_phase_exponent=0.2).on(q1),
        cirq.CX(q1, q3),
        cirq.CZ(q1, q3),
        cirq.CZ(q1, q3) ** 0.5,
        cirq.CCX(q2, q1, q3),
        cirq.CCZ(q3, q1, q2),
        cirq.rx(0.86).on(q3),
        cirq.ry(0.75).on(q3),
        cirq.rz(3.09).on(q3),
        cirq.QubitPermutationGate([1, 2, 0]).on(q1, q2, q3),
        cirq.WaitGate(cirq.Duration(nanos=20)).on(q3),
        cirq.XXPowGate(exponent=0.25, global_shift=-0.5).on(q1, q3),
        cirq.ZZPowGate(exponent=0.2, global_shift=-0.5).on(q1, q3),
        css.ZXPowGate(exponent=0.125, global_shift=-0.5)(q1, q3),
        css.AceCRPlusMinus(q1, q3),
        css.AceCRMinusPlus(q1, q3),
        css.AceCR("-+", sandwich_rx_rads=np.pi / 2)(q1, q3),
        css.ZZSwapGate(1.23).on(q1, q3),
        css.Barrier(2).on(q1, q3),
        css.Barrier(1).on(q3),
        css.AQTICCX(q1, q2, q3),
        css.StrippedCZGate(1.23).on(q1, q2),
        css.DDPowGate(exponent=0.7).on(q1, q2),
    )

    all_qubits: Sequence[cirq.Qid] = cirq.LineQubit.range(5)
    assert sm.converters.cirq_to_qiskit(cirq_circuit, all_qubits).num_qubits == 5
    assert sm.converters.cirq_to_qiskit(cirq_circuit, [q1, q2, q3]).num_qubits == 3

    # Test manual circuit comparison
    qiskit_circuit = qiskit.QuantumCircuit(5)
    qiskit_circuit.id(3)
    qiskit_circuit.x(3)
    qiskit_circuit.u(0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi, 1)
    qiskit_circuit.cx(1, 3)
    qiskit_circuit.cz(1, 3)
    qiskit_circuit.cp(np.pi / 2, 1, 3)
    qiskit_circuit.ccx(2, 1, 3)
    qiskit_circuit.ccz(3, 1, 2)
    qiskit_circuit.rx(0.86, 3)
    qiskit_circuit.ry(0.75, 3)
    qiskit_circuit.rz(3.09, 3)
    qiskit_circuit.append(qiskit.circuit.library.PermutationGate([2, 0, 1]), [1, 2, 3])
    qiskit_circuit.delay(20.0, 3, unit="ns")
    qiskit_circuit.rxx(np.pi / 4, 1, 3)
    qiskit_circuit.rzz(np.pi / 5, 1, 3)
    qiskit_circuit.rzx(np.pi / 8, 1, 3)
    qiskit_circuit.ecr(1, 3)
    qiskit_circuit.append(qss.AceCR("-+"), [1, 3])
    qiskit_circuit.append(qss.AceCR("-+", sandwich_rx_rads=np.pi / 2), [1, 3])
    qiskit_circuit.append(qss.ZZSwapGate(1.23), [1, 3])
    qiskit_circuit.barrier(1, 3)
    qiskit_circuit.barrier(3)
    qiskit_circuit.append(qss.AQTiCCXGate(), [1, 2, 3])
    qiskit_circuit.append(qss.StrippedCZGate(1.23), [1, 2])
    qiskit_circuit.append(qss.DDGate(0.7 * np.pi), [1, 2])
    assert qiskit_circuit == sm.converters.cirq_to_qiskit(cirq_circuit, all_qubits)
    assert cirq.approx_eq(cirq_circuit, sm.converters.qiskit_to_cirq(qiskit_circuit))

    phase = -0.1 * np.pi / 2  # 0.1 from the PhasedXZGate's x_exponent
    assert _operations_are_equivalent(qiskit_circuit, cirq_circuit, all_qubits, phase)

    # test cirq --> qiskit --> cirq conversion / recovery
    # transforming the qubits in cirq_circuit should not affect the resulting qiskit circuit as long
    # as the new qubits are passed to cirq_to_qiskit
    for new_qubits in (
        all_qubits,
        all_qubits[::-1],
        [*cirq.NamedQubit.range(3, prefix="qb_"), *cirq.GridQubit.rect(2, 1)],
    ):
        qubit_map: dict[cirq.Qid, cirq.Qid] = dict(zip(all_qubits, new_qubits))
        mapped_circuit = cirq_circuit.transform_qubits(qubit_map)

        assert qiskit_circuit == sm.converters.cirq_to_qiskit(mapped_circuit, new_qubits)

    # Circuit with measurements + resets
    cirq_circuit = cirq_circuit.copy()
    cirq_circuit += cirq.measure(q1, key="1")
    cirq_circuit += cirq.measure(q2, q3, key="2,3")
    cirq_circuit += cirq.measure(q1, key="cbit")
    cirq_circuit += cirq.measure(q2, key="4")
    cirq_circuit += cirq.reset(q3)
    cirq_circuit += cirq.measure(q3, key="4")

    # measurement keys are mapped to classical bit indices alphabetically,
    # i.e. ["1", "2,3", "cbit", "4"] -> [0, (1, 2), 4, 3]
    qiskit_circuit.add_register(qiskit.ClassicalRegister(5, "c"))
    qiskit_circuit.measure(1, 0)
    qiskit_circuit.measure(2, 1)
    qiskit_circuit.measure(3, 2)
    qiskit_circuit.measure(1, 4)
    qiskit_circuit.measure(2, 3)
    qiskit_circuit.reset(3)
    qiskit_circuit.measure(3, 3)

    assert qiskit_circuit == sm.converters.cirq_to_qiskit(cirq_circuit, all_qubits)


def test_cirq_to_qiskit_with_decompose() -> None:
    q0, q1, q2, q3 = qubits = cirq.LineQubit.range(4)
    cirq_circuit = cirq.Circuit(
        cirq.CCX(q0, q2, q3) ** 1.23,
        cirq.CCZ(q3, q1, q2) ** -0.3,
        cirq.CSWAP(q2, q0, q1),
        cirq.FSimGate(0.12, 0.34).on(q0, q3),
    )

    round_trip = sm.converters.qiskit_to_cirq(sm.converters.cirq_to_qiskit(cirq_circuit, qubits))

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(round_trip), cirq.unitary(cirq_circuit), atol=1e-8
    )


def test_cirq_to_qiskit_with_iterative_decompose() -> None:
    rng = np.random.default_rng()
    qubits = [cirq.LineQubit(int(i)) for i in rng.permutation(6)]
    cirq_circuit = cirq.testing.random_circuit(
        qubits[:3], 5, 1, gate_domain={cirq.X: 1, cirq.CX: 2, cirq.CCX: 3}
    )
    for _ in range(3):
        index = rng.choice(len(cirq_circuit) + 1)
        new_qubits = [qubits[i] for i in rng.permutation(len(qubits))]
        circuit_op = cirq.CircuitOperation(cirq_circuit.freeze()).with_qubit_mapping(
            dict(zip(qubits, new_qubits))
        )
        cirq_circuit.insert(index, circuit_op)

    unrolled_cirq_circuit = cirq.unroll_circuit_op(cirq_circuit, tags_to_check=None, deep=True)

    qiskit_circuit = sm.converters.cirq_to_qiskit(cirq_circuit, sorted(qubits))
    cirq.testing.assert_same_circuits(
        sm.converters.qiskit_to_cirq(qiskit_circuit),
        cirq.align_left(unrolled_cirq_circuit),
    )


def test_cirq_to_qiskit_invert_mask() -> None:
    q0, q1, q2, q3 = qubits = cirq.LineQubit.range(4)

    cirq_circuit = cirq.Circuit(
        cirq.measure(q3),
        cirq.measure(q2, invert_mask=(True,)),
        cirq.measure(q0, q1, invert_mask=(False, True)),
    )

    expected_qiskit_circuit = qiskit.QuantumCircuit(4, 4)
    expected_qiskit_circuit.x(1)
    expected_qiskit_circuit.x(2)
    expected_qiskit_circuit.measure([0, 1, 2, 3], [0, 1, 2, 3])

    assert sm.converters.cirq_to_qiskit(cirq_circuit, qubits) == expected_qiskit_circuit


def test_cirq_to_qiskit_classical_control() -> None:
    q0, q1, q2 = sympy.symbols("q0 q1 q2")
    sympy_cond = sympy.Add(q0, q1, q2)
    sympy_cond2 = sympy.Add(q0)
    sympy_cond3 = sympy.Eq(q1, 1)

    qubits = cirq.LineQubit.range(3)
    cirq_circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CX(*qubits[:2]), cirq.CX(*qubits[1:]))
    cirq_circuit += cirq.measure(qubits[0], key="q0")
    cirq_circuit += cirq.measure(qubits[1], key="q1")
    cirq_circuit += cirq.measure(qubits[2], key="q2")
    cirq_circuit += cirq.Z(qubits[0]).with_classical_controls(sympy_cond % 2)
    cirq_circuit += cirq.X(qubits[0]).with_classical_controls(sympy_cond2 % 2)
    cirq_circuit += cirq.X(qubits[1]).with_classical_controls(sympy_cond3)

    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    cr = qiskit.ClassicalRegister(3, "c")
    qc.add_register(cr)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)

    condition = qiskit.circuit.classical.expr.bit_xor(cr[0], cr[1])
    condition = qiskit.circuit.classical.expr.bit_xor(condition, cr[2])
    condition2 = (cr[0], 1)
    condition3 = (cr[1], 1)
    with qc.if_test(condition):
        qc.z(0)
    with qc.if_test(condition2):
        qc.x(0)
    with qc.if_test(condition3):
        qc.x(1)

    assert sm.converters.cirq_to_qiskit(cirq_circuit, sorted(cirq_circuit.all_qubits())) == qc

    q0, q1, q2 = sympy.symbols("q0 q1 q2")
    sympy_cond = sympy.Mul(q0, q1, q2)

    qubits = cirq.LineQubit.range(3)
    cirq_circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CX(*qubits[:2]), cirq.CX(*qubits[1:]))
    cirq_circuit += cirq.measure(qubits[0], key="q0")
    cirq_circuit += cirq.measure(qubits[1], key="q1")
    cirq_circuit += cirq.measure(qubits[2], key="q2")
    cirq_circuit += cirq.Z(qubits[0]).with_classical_controls(sympy_cond % 2)

    with pytest.raises(gss.SuperstaqException, match="We don't currently support"):
        sm.converters.cirq_to_qiskit(cirq_circuit, sorted(cirq_circuit.all_qubits()))

    q0, q1 = sympy.symbols("q0 q1")
    sympy_cond = sympy.Add(q0, q1)
    sympy_cond2 = sympy.Eq(q1, 1)

    qubits = cirq.LineQubit.range(3)
    cirq_circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CX(*qubits[:2]))
    cirq_circuit += cirq.measure(qubits[0], key="q0")
    cirq_circuit += cirq.measure(qubits[1], key="q1")
    cirq_circuit += (
        cirq.Z(qubits[0])
        .with_classical_controls(sympy_cond % 2)
        .with_classical_controls(sympy_cond2)
    )

    with pytest.raises(
        gss.SuperstaqException,
        match=re.escape(
            "We don't currently support multiple layers of "
            "classical control on a single operation.",
        ),
    ):
        sm.converters.cirq_to_qiskit(cirq_circuit, sorted(cirq_circuit.all_qubits()))

    qubits = cirq.LineQubit.range(3)
    cirq_circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CX(*qubits[:2]))
    cirq_circuit += cirq.measure(qubits[0], key="q0")
    cirq_circuit += cirq.measure(qubits[1], key="q1")
    cirq_circuit += cirq.Z(qubits[0]).with_classical_controls("q1")

    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    cr = qiskit.ClassicalRegister(2, "c")
    qc.add_register(cr)
    qc.measure(0, 0)
    qc.measure(1, 1)

    condition = (cr[1], 1)
    with qc.if_test(condition):
        qc.z(0)

    assert sm.converters.cirq_to_qiskit(cirq_circuit, sorted(cirq_circuit.all_qubits())) == qc


def test_cirq_qubits_to_qiskit() -> None:
    qubits = [cirq.NamedQubit("0+"), cirq.NamedQubit("0-")]
    cirq_circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CX(*qubits))

    qr1 = qiskit.QuantumRegister(1, "0+")
    qr2 = qiskit.QuantumRegister(1, "0-")
    qc = qiskit.QuantumCircuit(qr1, qr2)
    qc.h(qr1[0])
    qc.cx(qr1[0], qr2[0])

    assert sm.converters.cirq_to_qiskit(cirq_circuit, sorted(cirq_circuit.all_qubits())) == qc
