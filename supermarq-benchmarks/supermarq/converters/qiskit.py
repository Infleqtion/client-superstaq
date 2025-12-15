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

import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import cirq
import cirq_superstaq as css
import general_superstaq as gss
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
import qiskit
import qiskit_superstaq as qss
import sympy

CIRCUIT_TYPE = TypeVar("CIRCUIT_TYPE", bound=cirq.AbstractCircuit)

# Gates with no arguments (aside from possibly `label`), such that qiskit_gate() <==> cirq_gate.
# All unitaries must have the same global phase.
_qiskit_static_gates: dict[type[qiskit.circuit.Instruction], cirq.Gate] = {
    qiskit.circuit.library.IGate: cirq.I,
    qiskit.circuit.library.HGate: cirq.H,
    qiskit.circuit.library.XGate: cirq.X,
    qiskit.circuit.library.YGate: cirq.Y,
    qiskit.circuit.library.ZGate: cirq.Z,
    qiskit.circuit.library.SGate: cirq.S,
    qiskit.circuit.library.TGate: cirq.T,
    qiskit.circuit.library.SXGate: cirq.X**0.5,
    qiskit.circuit.library.SdgGate: cirq.S**-1,
    qiskit.circuit.library.TdgGate: cirq.T**-1,
    qiskit.circuit.library.SXdgGate: cirq.X**-0.5,
    qiskit.circuit.library.SwapGate: cirq.SWAP,
    qiskit.circuit.library.iSwapGate: cirq.ISWAP,
    qiskit.circuit.library.ECRGate: css.AceCR("+-"),
    qss.custom_gates.iXGate: css.ops.qubit_gates.IX,
    qss.custom_gates.iXdgGate: css.ops.qubit_gates.IX**-1,
    qiskit.circuit.Reset: cirq.ResetChannel(),
}

# Static gates mapped from cirq and qiskit
_cirq_static_gates: dict[cirq.Gate, qiskit.circuit.Gate] = {
    cirq.I: qiskit.circuit.library.IGate,
    cirq.X: qiskit.circuit.library.XGate,
    cirq.Y: qiskit.circuit.library.YGate,
    cirq.Z: qiskit.circuit.library.ZGate,
    cirq.H: qiskit.circuit.library.HGate,
    cirq.S: qiskit.circuit.library.SGate,
    cirq.S**-1: qiskit.circuit.library.SdgGate,
    cirq.X**0.5: qiskit.circuit.library.SXGate,
    cirq.CX: qiskit.circuit.library.CXGate,
    cirq.Y.controlled(1): qiskit.circuit.library.CYGate,
    cirq.CZ: qiskit.circuit.library.CZGate,
    cirq.SWAP: qiskit.circuit.library.SwapGate,
    cirq.ISWAP: qiskit.circuit.library.iSwapGate,
    cirq.CSWAP: qiskit.circuit.library.CSwapGate,
    cirq.CCX: qiskit.circuit.library.CCXGate,
    cirq.CCZ: qiskit.circuit.library.CCZGate,
    css.AceCR(): qiskit.circuit.library.ECRGate,
    css.AQTITOFFOLI: qss.AQTiToffoliGate,
    cirq.ResetChannel(): qiskit.circuit.Reset,
}

# Rotation gates with exactly one argument (aside from possibly `label`), with one-to-one mapping
# between `qiskit_gate(z * pi)` and `cirq_gate**z`. Unitaries must have the same global phase.
_qiskit_pow_gates: dict[type[qiskit.circuit.Instruction], cirq.Gate] = {
    qiskit.circuit.library.PhaseGate: cirq.Z,
    qiskit.circuit.library.U1Gate: cirq.Z,
    qiskit.circuit.library.RXGate: cirq.rx(np.pi),
    qiskit.circuit.library.RYGate: cirq.ry(np.pi),
    qiskit.circuit.library.RZGate: cirq.rz(np.pi),
    qiskit.circuit.library.RXXGate: cirq.XXPowGate(global_shift=-0.5),
    qiskit.circuit.library.RYYGate: cirq.YYPowGate(global_shift=-0.5),
    qiskit.circuit.library.RZZGate: cirq.ZZPowGate(global_shift=-0.5),
    qiskit.circuit.library.RZXGate: css.ZXPowGate(global_shift=-0.5),
    qss.DDGate: css.DDPowGate(),
}

# Single-parameter rotation gates mapped from cirq to qiskit
_cirq_pow_gates: dict[type[cirq.EigenGate], type[qiskit.circuit.Gate]] = {
    cirq.XPowGate: qiskit.circuit.library.RXGate,
    cirq.YPowGate: qiskit.circuit.library.RYGate,
    cirq.ZPowGate: qiskit.circuit.library.RZGate,
    cirq.CZPowGate: qiskit.circuit.library.CPhaseGate,
    cirq.XXPowGate: qiskit.circuit.library.RXXGate,
    cirq.YYPowGate: qiskit.circuit.library.RYYGate,
    cirq.ZZPowGate: qiskit.circuit.library.RZZGate,
    css.ZXPowGate: qiskit.circuit.library.RZXGate,
    css.DDPowGate: qss.DDGate,
}

# Other parameterized gates, such that `cirq_gate == _qiskit_param_gates[type(inst)](inst)`.
# Unitaries must have the same global phase.
_qiskit_param_gates: dict[
    type[qiskit.circuit.Instruction], Callable[[qiskit.circuit.Instruction], cirq.Gate]
] = {
    qiskit.circuit.library.RGate: lambda inst: css.RGate(*inst.params),
    qiskit.circuit.library.Barrier: lambda inst: css.Barrier(inst.num_qubits),
    qss.ZZSwapGate: lambda inst: css.ZZSwapGate(*inst.params),
    qss.AceCR: lambda inst: css.AceCR(*inst.params),
    qiskit.circuit.Delay: lambda inst: cirq.WaitGate(_get_cirq_duration(inst.duration, inst.unit)),
    qss.StrippedCZGate: lambda inst: css.StrippedCZGate(*inst.params),
    qiskit.circuit.library.PermutationGate: lambda inst: cirq.QubitPermutationGate(
        [list(inst.pattern).index(i) for i in range(inst.num_qubits)]
    ),
}

# Parametrized gates mapped from cirq to qiskit
_cirq_param_gates: dict[type[cirq.Gate], Callable[[Any], qiskit.QuantumCircuit]] = {
    cirq.circuits.qasm_output.QasmUGate: lambda cirq_gate: qiskit.circuit.library.UGate(
        cirq_gate.theta * np.pi, cirq_gate.phi * np.pi, cirq_gate.lmda * np.pi
    ),
    cirq.PhasedXZGate: lambda cirq_gate: qiskit.circuit.library.UGate(
        cirq_gate.x_exponent * np.pi,
        (cirq_gate.z_exponent + cirq_gate.axis_phase_exponent - 0.5) * np.pi,
        (0.5 - cirq_gate.axis_phase_exponent) * np.pi,
    ),
    css.RGate: lambda cirq_gate: qiskit.circuit.library.RGate(cirq_gate.theta, cirq_gate.phi),
    css.ParallelRGate: lambda cirq_gate: qiskit.circuit.library.GR(
        cirq_gate.num_qubits(), cirq_gate.theta, cirq_gate.phi
    ),
    cirq.PhasedXPowGate: lambda cirq_gate: qiskit.circuit.library.RGate(
        cirq_gate.exponent * np.pi, cirq_gate.phase_exponent * np.pi
    ),
    css.Barrier: lambda cirq_gate: qiskit.circuit.library.Barrier(cirq_gate.num_qubits()),
    css.ZZSwapGate: lambda cirq_gate: qss.ZZSwapGate(cirq_gate.theta),
    css.AceCR: lambda cirq_gate: qss.AceCR(cirq_gate.rads, cirq_gate.sandwich_rx_rads),
    css.StrippedCZGate: lambda cirq_gate: qss.StrippedCZGate(cirq_gate.rz_rads),
    css.ParallelGates: lambda cirq_gate: qss.ParallelGates(
        *[cirq_gate_to_qiskit_gate(gate) for gate in cirq_gate.component_gates]
    ),
    cirq.MatrixGate: lambda cirq_gate: qiskit.circuit.library.UnitaryGate(
        swap_endianness(cirq.unitary(cirq_gate)), label=cirq_gate._name
    ),
    cirq.QubitPermutationGate: lambda cirq_gate: qiskit.circuit.library.PermutationGate(
        [cirq_gate.permutation.index(i) for i in range(cirq_gate.num_qubits())]
    ),
}

# Gate types in this tuple will be decomposed using their .definition, rather than in cirq (because
# some qiskit gates are used for a particular decomposition)
_keep_qiskit_definition = (
    qiskit.circuit.library.CSwapGate,
    qiskit.circuit.library.C3SXGate,
    qiskit.circuit.library.C3XGate,
    qiskit.circuit.library.C4XGate,
    qiskit.circuit.library.RCCXGate,
    qiskit.circuit.library.MCXGrayCode,
    qiskit.circuit.library.MCXRecursive,
    qiskit.circuit.library.MCXVChain,
    qiskit.circuit.library.MCMTGate,
)

# Gates classes which are known to always be handled correctly. Everything here must satisfy
# >>> inst == inst.base_gate.control(inst.num_ctrl_qubits, ctrl_state=inst.ctrl_state)
# (which is not true of e.g. qiskit.circuit.library.CUGate). Populating this is not strictly
# necessary, but it offers a speedup by bypassing the equality check for common gate types
_known_qiskit_controlled_gate_types = (
    qiskit.circuit.library.CXGate,
    qiskit.circuit.library.CYGate,
    qiskit.circuit.library.CZGate,
    qiskit.circuit.library.CHGate,
    qiskit.circuit.library.CSGate,
    qiskit.circuit.library.CSdgGate,
    qiskit.circuit.library.CPhaseGate,
    qiskit.circuit.library.CCZGate,
    qiskit.circuit.library.CSXGate,
    qiskit.circuit.library.CRXGate,
    qiskit.circuit.library.CRYGate,
    qiskit.circuit.library.CRZGate,
    qiskit.circuit.library.CU1Gate,
    qiskit.circuit.library.CU3Gate,
    qiskit.circuit.library.CCXGate,
    qiskit.circuit.library.MCXGate,
    qiskit.circuit.library.MCU1Gate,
    qiskit.circuit.library.MCPhaseGate,
    qss.custom_gates.iCCXGate,
    qss.custom_gates.iCCXdgGate,
    qss.AQTiCCXGate,
)


def swap_endianness(matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Change matrix from cirq-endian and qiskit-endian (and vice versa)."""
    n = int(np.log2(len(matrix)))
    assert np.shape(matrix) == (2**n, 2**n)

    axes = (*reversed(range(n)), *reversed(range(n, 2 * n)))
    return np.transpose(matrix.reshape((2,) * (2 * n)), axes).reshape(2**n, 2**n)


def cirq_to_qiskit(
    cirq_circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
) -> qiskit.QuantumCircuit:
    """Converts `cirq.Circuit` into a `qiskit.QuantumCircuit` with a single qubit register.

    Note: Qubits in the qiskit circuit are ordered according to the given qubits sequence
    (i.e. n-th qubit in `qiskit_circuit.qubits` = n-th qubit in sequence).

    Args:
        cirq_circuit: The circuit to convert into `qiskit`.
        qubits: The qubits belonging to `cirq_circuit`.

    Returns:
        The `qiskit` equivalent of `cirq_circuit`.
    """
    cirq_qubit_to_index = {qubit: index for index, qubit in enumerate(qubits)}

    qiskit_circuit = cirq_qubits_to_qiskit(qubits)

    # assign measurement keys to classical bits in alphabetical order
    measurement_key_to_indices = {}
    if cirq_circuit.has_measurements():
        measurement_sizes = {
            cirq.measurement_key_name(op): len(op.qubits)
            for _, op in cirq_circuit.findall_operations(cirq.is_measurement)
        }

        num_clbits = 0
        for key in sorted(cirq_circuit.all_measurement_key_names()):
            measurement_key_to_indices[key] = [
                num_clbits + i for i in range(measurement_sizes[key])
            ]
            num_clbits += measurement_sizes[key]

        cr = qiskit.ClassicalRegister(sum(measurement_sizes.values()), "c")
        qiskit_circuit.add_register(cr)

        cirq_circuit = decompose_measurement_inversions(cirq_circuit)

    for op in cirq_circuit.all_operations():
        qubit_indices = [cirq_qubit_to_index[q] for q in op.qubits]
        qargs: Sequence[int | list[int]] = qubit_indices
        cargs: Sequence[int | list[int]] = []

        if isinstance(op.gate, cirq.MeasurementGate):
            qiskit_gate = qiskit.circuit.Measure()
            qargs = [qubit_indices]
            cargs = [measurement_key_to_indices[op.gate.key]]

        elif isinstance(op.gate, cirq.IdentityGate) and not isinstance(op.gate, css.Barrier):
            # Special case for multi-qubit IdentityGate (no qiskit equivalent)
            qiskit_gate = qiskit.circuit.library.IGate()
            qargs = [qubit_indices]

        elif op.classical_controls:
            condition = cirq_classical_control_to_qiskit(op, measurement_key_to_indices, cr)

            with qiskit_circuit.if_test(condition):
                cirq_gate = op.without_classical_controls().gate
                assert cirq_gate
                qiskit_gate = cirq_gate_to_qiskit_gate(cirq_gate)
                qiskit_circuit.append(qiskit_gate, qubit_indices)
            # Remove pragma after removal of Python 3.9 support
            continue  # pragma: no cover

        elif op.gate:
            qiskit_gate = cirq_gate_to_qiskit_gate(op.gate)

        else:
            # Handle circuit ops
            qiskit_gate = _cirq_op_to_custom_gate(op)

        qiskit_circuit.append(qiskit_gate, qargs, cargs)
    return qiskit_circuit


@cirq.transformer
def decompose_measurement_inversions(
    circuit: CIRCUIT_TYPE, context: cirq.TransformerContext | None = None
) -> CIRCUIT_TYPE:
    """Decompose inverted measurements by replacing their `invert_mask` with explicit X gates.

    This is useful after running some cirq transformers (namely `cirq.eject_phased_paulis`), which
    will absorb Pauli gates into measurements by changing their `invert_mask`.

    Args:
        circuit: The circuit to be transformed.
        context: An optional `cirq.TransformerContext` containing configuration and logging
            information.

    Returns:
        An equivalent circuit with X gates in place of inversion masks.
    """

    def _map_fn(op: cirq.Operation, _: int) -> Iterator[cirq.Operation]:
        if isinstance(op.gate, cirq.MeasurementGate) and any(op.gate.invert_mask):
            inverted_bits = [i for i, x in enumerate(op.gate.invert_mask) if x]
            for i in inverted_bits:
                yield cirq.X(op.qubits[i])

            # Construct a new measurement op the with default `invert_mask`. This doesn't use
            # `gate.with_bits_flipped` because of https://github.com/quantumlib/Cirq/issues/6093.
            gate = cirq.MeasurementGate(
                cirq.num_qubits(op), key=op.gate.mkey, confusion_map=op.gate.confusion_map
            )
            op = gate.on(*op.qubits)
        yield op

    return cirq.map_operations_and_unroll(
        circuit,
        _map_fn,
        deep=context.deep if context else False,
        tags_to_ignore=context.tags_to_ignore if context else (),
    )


def cirq_qubits_to_qiskit(qubits: Sequence[cirq.Qid]) -> qiskit.QuantumCircuit:
    """Converts cirq qubits to qiskit qubits.

    Args:
        qubits: A sequence of `cirq.Qid` s.

    Returns:
        A `qiskit.QuantumCircuit` object with the appropriate qubits.
    """
    if all(isinstance(q, cirq.NamedQubit) for q in qubits):
        qiskit_qubits = [qiskit.QuantumRegister(1, str(q)) for q in qubits]
        return qiskit.QuantumCircuit(*qiskit_qubits)

    qiskit_qubits = qiskit.QuantumRegister(len(qubits), "q")
    return qiskit.QuantumCircuit(qiskit_qubits)


def cirq_classical_control_to_qiskit(
    op: cirq.Operation,
    measurement_key_to_indices: dict[str, list[int]],
    cr: qiskit.ClassicalRegister,
) -> qiskit.circuit.classical.expr.Expr | tuple[qiskit.circuit.Clbit, int]:
    """Converts cirq classical control to qiskit classical control. Cirq uses sympy for its
    classical logic, qiskit uses its own classical expressions.

    Args:
        op: The cirq operation that is being classically controlled
        measurement_key_to_indices: Dictionary that maps cirq measurement keys to qiskit
        `Clbit` indices.
        cr: The qiskit classical register that controls the operations

    Returns:
        A qiskit condition for classical control.
    """
    condition = None
    if len(list(op.classical_controls)) > 1:
        raise gss.SuperstaqException(
            "We don't currently support multiple layers of classical control on a single operation."
        )

    cirq_condition = next(iter(op.classical_controls))
    if isinstance(cirq_condition, cirq.SympyCondition):
        sympy_expr = cirq_condition.expr
        p, q = sympy_expr.args
        if isinstance(sympy_expr, sympy.Mod):
            if not p.args:
                assert isinstance(p, sympy.Symbol)
                condition = (cr[measurement_key_to_indices[p.name][0]], 1)
            elif isinstance(p, sympy.Add) and isinstance(q, sympy.Integer) and int(q) == 2:
                symbols = p.args
                assert isinstance(symbols[0], sympy.Symbol)
                assert isinstance(symbols[1], sympy.Symbol)
                condition = qiskit.circuit.classical.expr.bit_xor(
                    cr[measurement_key_to_indices[symbols[0].name][0]],
                    cr[measurement_key_to_indices[symbols[1].name][0]],
                )
                for idx in list(p.args)[2:]:
                    assert isinstance(idx, sympy.Symbol)
                    condition = qiskit.circuit.classical.expr.bit_xor(
                        condition, cr[measurement_key_to_indices[idx.name][0]]
                    )

        elif isinstance(sympy_expr, sympy.Eq):
            assert isinstance(p, sympy.Symbol)
            condition = (cr[measurement_key_to_indices[p.name][0]], 1)

        if not condition:
            raise gss.SuperstaqException(
                f"We don't currently support {sympy_expr} in our qiskit classical control flow."
            )

    elif isinstance(cirq_condition, cirq.KeyCondition):
        p = cirq_condition.keys[0]
        assert isinstance(p, cirq.MeasurementKey)
        condition = (cr[measurement_key_to_indices[p.name][0]], 1)

    return condition


def _cirq_op_to_custom_gate(op: cirq.Operation) -> qiskit.circuit.Gate:
    decomp = cirq.decompose_once(op, default=None)
    if decomp is None:
        raise NotImplementedError(f"Unable to convert {op} to qiskit.")

    subcircuit = cirq_to_qiskit(cirq.Circuit(decomp), op.qubits)
    return subcircuit.to_instruction()


def cirq_gate_to_qiskit_gate(cirq_gate: cirq.Gate) -> qiskit.circuit.Gate:
    """Convert a single `cirq.Gate` to a qiskit Gate."""
    if cirq_gate in _cirq_static_gates:
        return _cirq_static_gates[cirq_gate]()

    for gate_type in _cirq_param_gates:
        if isinstance(cirq_gate, gate_type):
            return _cirq_param_gates[gate_type](cirq_gate)

    for gate_type in _cirq_pow_gates:
        if isinstance(cirq_gate, gate_type):
            return _cirq_pow_gates[gate_type](cirq_gate.exponent * np.pi)

    if isinstance(cirq_gate, cirq.ControlledGate):
        ctrl_state = "".join("0" if 0 in val else "1" for val in cirq_gate.control_values)[::-1]
        base_gate = cirq_gate_to_qiskit_gate(cirq_gate.sub_gate)
        return base_gate.control(cirq_gate.num_controls(), ctrl_state=ctrl_state)

    if isinstance(cirq_gate, cirq.WaitGate):
        return qiskit.circuit.Delay(cirq_gate.duration.total_nanos(), unit="ns")

    # Fall back on decomposition
    qubits = cirq.LineQid.for_gate(cirq_gate)
    return _cirq_op_to_custom_gate(cirq_gate(*qubits))


def qiskit_to_cirq(
    qiskit_circuit: qiskit.QuantumCircuit, keep_global_phase: bool = False
) -> cirq.Circuit:
    """Converts a `qiskit.QuantumCircuit` to a `cirq.Circuit`, preserving global phase.

    Args:
        qiskit_circuit: The circuit to convert into `cirq`.
        keep_global_phase: Boolean flag to preserve the global phase of the circuit. Defaults to
            `False`.

    Returns:
        The `cirq` equivalent circuit of `qiskit_circuit`.
    """
    qubits = [cirq.LineQubit(i) for i in range(qiskit_circuit.num_qubits)]
    measurement_keys: list[str] = []
    for creg in qiskit_circuit.cregs:
        pad = len(f"{creg.size - 1}")
        measurement_keys += [f"{creg.name}_{i:0>{pad}}" for i in range(creg.size)]
    cirq_circuit, phase = _handle_qiskit_circuit(qiskit_circuit, qubits, measurement_keys)

    if keep_global_phase and not cirq.all_near_zero_mod(phase, 2 * np.pi):
        cirq_circuit += cirq.global_phase_operation(np.exp(1j * phase))

    assert cirq_circuit.all_qubits().issubset(qubits)
    assert set(cirq_circuit.all_measurement_key_names()).issubset(measurement_keys)
    return cirq_circuit


def qiskit_gate_to_cirq_gate(inst: qiskit.circuit.Instruction) -> cirq.Gate | None:
    """Convert a single qiskit instruction to a single cirq.Gate, or return None."""
    if not isinstance(inst, qiskit.circuit.Instruction):
        return None

    if cirq_gate := _qiskit_static_gates.get(inst.base_class):
        return cirq_gate

    if cirq_gate := _qiskit_pow_gates.get(inst.base_class):
        return cirq_gate ** (inst.params[0] / np.pi)

    if cirq_gate_fn := _qiskit_param_gates.get(inst.base_class):
        return cirq_gate_fn(inst)

    return None


def _handle_qiskit_circuit(
    qiskit_circuit: qiskit.QuantumCircuit,
    cirq_qubits: Sequence[cirq.Qid],
    measurement_keys: Sequence[str],
) -> tuple[cirq.Circuit, float]:
    """Convert a qiskit QuantumCircuit to a cirq Circuit and a global phase."""
    assert len(cirq_qubits) >= qiskit_circuit.num_qubits
    assert len(measurement_keys) >= qiskit_circuit.num_clbits

    qubit_map = dict(zip(qiskit_circuit.qubits, cirq_qubits))
    clbit_map = dict(zip(qiskit_circuit.clbits, measurement_keys))

    cirq_circuit = cirq.Circuit()
    global_phase = qiskit_circuit.global_phase

    for inst in qiskit_circuit:
        cirq_qubits = [qubit_map[q] for q in inst.qubits]
        measurement_keys = [clbit_map[c] for c in inst.clbits]

        # Check qiskit classical controls first, if any:
        if inst_condition := getattr(inst.operation, "condition", None):
            classical, val = inst_condition

            if isinstance(classical, qiskit.circuit.Clbit):
                classical = [classical]

            conditions: list[str | cirq.MeasurementKey | cirq.Condition | sympy.Expr] = []
            for i, clbit in enumerate(classical):
                if val >> i & 1:
                    conditions.append(clbit_map[clbit])
                else:
                    conditions.append(1 - sympy.var(clbit_map[clbit]))

            conditional_circ = cirq.Circuit()
            for conditional_block in inst.operation.blocks:
                sub_cond_circ, sub_phase = _handle_qiskit_circuit(
                    conditional_block, cirq_qubits, measurement_keys
                )
                conditional_circ += sub_cond_circ
                global_phase += sub_phase
            cirq_ops = [
                op.with_classical_controls(*conditions) for op in conditional_circ.all_operations()
            ]
        else:
            cirq_op, phase = _handle_qiskit_inst(inst.operation, cirq_qubits, measurement_keys)
            cirq_ops = [cirq_op]
            global_phase += phase

        for cirq_op in cirq_ops:
            is_circuit_operation = isinstance(cirq_op, cirq.CircuitOperation)

            tags = []
            if inst.operation.label:
                tags.append(inst.operation.label)

            # Unroll CircuitOperations before appending, unless tagged with "no_compile"
            if is_circuit_operation and "no_compile" not in tags:
                cirq_circuit += [op.with_tags(*tags) for op in cirq.decompose_once(cirq_op)]
            else:
                cirq_circuit += cirq_op.with_tags(*tags)

    global_phase %= 2 * np.pi

    return cirq_circuit, global_phase


def _handle_qiskit_inst(
    inst: qiskit.circuit.Instruction,
    cirq_qubits: Sequence[cirq.Qid],
    measurement_keys: Sequence[str],
) -> tuple[cirq.Operation, float]:
    _parameter_expressions_to_float(inst)

    # TODO: for QASM conversion
    # if type(inst) in (qiskit.circuit.Gate, qiskit.circuit.Instruction):
    #     inst = rewrite_generic_gate(inst)  # noqa: ERA001

    if issubclass(inst.base_class, _keep_qiskit_definition):
        pass

    elif inst.base_class is qiskit.circuit.Measure:
        key = ",".join(measurement_keys)
        return cirq.measure(*cirq_qubits, key=key), 0.0

    elif cirq_gate := qiskit_gate_to_cirq_gate(inst):
        return cirq_gate.on(*cirq_qubits), 0.0

    elif inst.base_class is qiskit.circuit.library.UGate:
        cirq_gate, phase = _handle_qiskit_u_gate(*inst.params)
        return cirq_gate.on(*cirq_qubits), phase

    elif issubclass(inst.base_class, qiskit.circuit.ControlledGate):
        if cirq_op := _handle_qiskit_controlled_op(inst, cirq_qubits):
            return cirq_op, 0.0

    if inst.definition is not None and inst.base_class is not qiskit.circuit.library.UnitaryGate:
        return _handle_qiskit_definition(inst, cirq_qubits, measurement_keys)

    # Finally fall back on a big MatrixGate
    if hasattr(inst, "__array__"):
        mat = inst.to_matrix()
        # (qubit order reversed to convert from qiskit-endian to cirq-endian)
        return cirq.MatrixGate(mat).on(*reversed(cirq_qubits)), 0.0

    raise gss.SuperstaqException(
        f"We don't know what to do with definition-less {inst.base_class.__name__} {inst.name}("
        + ",".join(map(str, inst.params))
        + ")."
    )


def _handle_qiskit_u_gate(theta: float, phi: float, lam: float) -> tuple[cirq.Gate, float]:
    """Qiskit's UGate does not have the same global phase as either cirq's QasmUGate or
    PhasedXZGate.
    """
    x_exponent = theta / np.pi
    z_exponent = (phi + lam) / np.pi
    axis_phase_exponent = 0.5 - lam / np.pi
    global_shift = -theta / (2 * np.pi)

    if cirq.all_near_zero_mod(x_exponent, 2):
        return cirq.Z**z_exponent, 0.0

    if cirq.all_near_zero_mod(z_exponent, 2):
        return (
            cirq.PhasedXPowGate(
                exponent=x_exponent,
                phase_exponent=axis_phase_exponent,
                global_shift=-0.5,
            ),
            0.0,
        )

    if np.isclose(x_exponent % 2, 1):
        return (
            cirq.PhasedXZGate(
                x_exponent=x_exponent,
                z_exponent=z_exponent + 2 * global_shift,
                axis_phase_exponent=axis_phase_exponent - global_shift,
            ),
            0.0,
        )

    return (
        cirq.PhasedXZGate(
            x_exponent=x_exponent,
            z_exponent=z_exponent,
            axis_phase_exponent=axis_phase_exponent,
        ),
        global_shift * np.pi,
    )


def _handle_qiskit_controlled_op(
    inst: qiskit.circuit.ControlledGate,
    cirq_qubits: Sequence[cirq.Qid],
) -> cirq.Operation | None:
    """Convert a `qiskit.circuit.ControlledGate` to a `cirq.ControlledOperation`.

    This requires that:
        `inst == inst.base_gate.control(inst.num_ctrl_qubits, ctrl_state=inst.ctrl_state)`

    `CUGate` is also supported here as a special case even though this condition is not met. For
    any other `ControlledGate` not meeting this condition, returns `None`.
    """
    if (
        inst.base_class in _known_qiskit_controlled_gate_types
        or inst.base_class is qiskit.circuit.library.CUGate
        or inst == inst.base_gate.control(inst.num_ctrl_qubits, ctrl_state=inst.ctrl_state)
    ):
        num_ctrl_qubits = inst.num_ctrl_qubits
        target_qubits = cirq_qubits[num_ctrl_qubits:]
        control_qubits = cirq_qubits[:num_ctrl_qubits]
        control_values = [inst.ctrl_state >> i & 1 for i in range(num_ctrl_qubits)]

        base_op, phase = _handle_qiskit_inst(inst.base_gate, target_qubits, [])

        if all(control_values):
            controlled_op = base_op.controlled_by(*control_qubits, control_values=control_values)
        else:
            # `gate.controlled()` sometimes has strange behavior for non-default control values,
            # e.g. creating a controlled-`CZPowGate` instead of a doubly-controlled `ZPowGate`
            controlled_op = cirq.ControlledOperation(control_qubits, base_op, control_values)

        if isinstance(base_op, cirq.CircuitOperation):
            cirq_ops = cirq.decompose_once(controlled_op)
        else:
            cirq_ops = [controlled_op]

        # handle CUGate's extra phase parameter
        if inst.base_class is qiskit.circuit.library.CUGate:
            phase += inst.params[3]

        # any "global" phase is local after adding controls
        if not cirq.all_near_zero_mod(phase, 2 * np.pi):
            phase_op = cirq.global_phase_operation(np.exp(1j * phase))
            cirq_ops.append(phase_op.controlled_by(*control_qubits, control_values=control_values))

        # If controlled op requires more than one cirq operation, construct a new CircuitOperation
        # which includes the control (all the mapping was handled during the decomposition of
        # base_gate so no need for qubit_map or measurement_key_map here)
        if len(cirq_ops) > 1:
            return cirq.CircuitOperation(cirq.Circuit(cirq_ops).freeze())

        return cirq_ops[0]

    return None


def _handle_qiskit_definition(
    inst: qiskit.circuit.Instruction,
    cirq_qubits: Sequence[cirq.Qid],
    measurement_keys: Sequence[str],
) -> tuple[cirq.Operation, float]:
    """Convert a qiskit Instruction to a cirq Operation using its .definition."""
    cirq_circuit, phase = _handle_qiskit_circuit(inst.definition, cirq_qubits, measurement_keys)

    if len(cirq_circuit) == 1 and not cirq_circuit.has_measurements():
        # intercept ParallelGates
        if inst.name.startswith("parallel_"):
            return css.parallel_gates_operation(*cirq_circuit.all_operations()), phase

        # intercept ParallelRGate
        elif re.match(r"(gate_)?GR[\d_]*\b", inst.name):
            if parallel_op := _intercept_qiskit_parallel_r_gate(cirq_circuit):
                return parallel_op, phase

    # wrap in a CircuitOperation
    return cirq.CircuitOperation(cirq_circuit.freeze()), phase


def _intercept_qiskit_parallel_r_gate(cirq_circuit: cirq.Circuit) -> cirq.Operation | None:
    """If circuit is expressible as a single ParallelRGate, return that; otherwise return None."""
    rgate: css.RGate | None = None
    qubits: list[cirq.Qid] = []
    for op in cirq_circuit.all_operations():
        if not rgate and isinstance(op.gate, css.RGate):
            rgate = op.gate
        elif op.gate != rgate or op.qubits[0] in qubits:
            return None
        qubits.extend(op.qubits)

    if rgate:
        return css.ParallelRGate(rgate.theta, rgate.phi, len(qubits)).on(*qubits)

    return None


def _get_cirq_duration(duration: float, unit: str) -> cirq.Duration:
    unit_conversions = {
        "ps": cirq.Duration(picos=1),
        "ns": cirq.Duration(nanos=1),
        "us": cirq.Duration(micros=1),
        "ms": cirq.Duration(millis=1),
        "s": cirq.Duration(millis=1000),
    }

    if unit not in unit_conversions:
        raise gss.SuperstaqException(f"We only support Delay gates with SI units (got '{unit}')")

    return duration * unit_conversions[unit]


def _parameter_expressions_to_float(inst: qiskit.circuit.Instruction) -> None:
    """Takes in a qiskit instruction (`inst`) and tries to update it such that all
    parameter expressions have been turned into floats.

    Args:
        inst: The qiskit instruction to update.

    Raises:
        SuperstaqException: When converting parameter expressions to `float` fails.
    """
    for i, param in enumerate(inst.params):
        if isinstance(param, qiskit.circuit.ParameterExpression):
            try:
                inst.params[i] = float(param)
            except (TypeError, ValueError):
                try:
                    val = complex(param)
                    assert np.isclose(val.imag, 0)
                    inst.params[i] = val.real
                except (AssertionError, TypeError, ValueError):
                    raise gss.SuperstaqException(
                        "Can't convert parameterized unbounded qiskit circuits. Please let us know "
                        "if you'd like this feature."
                    )
