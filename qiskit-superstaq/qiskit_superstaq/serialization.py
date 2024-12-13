from __future__ import annotations

import io
import json
import re
import warnings
from collections.abc import Callable, Sequence
from typing import TypeVar

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit.qpy

import qiskit_superstaq as qss

T = TypeVar("T")

# Version to use for serialization. Deserialization can't be done with a QPY version older than that
# used for serialization, so using a slightly lower version prevents us from having to force users
# to update Qiskit the moment we do.
# Should always be QPY_COMPATIBILITY_VERSION <= QPY_SERIALIZATION_VERSION <= QPY_VERSION
QPY_SERIALIZATION_VERSION = 11

# Custom gate types to resolve when deserializing circuits
# MSGate included as a workaround for https://github.com/Qiskit/qiskit/issues/11378
_custom_gates_by_name: dict[str, type[qiskit.circuit.Instruction]] = {
    "acecr": qss.custom_gates.AceCR,
    "dd": qss.custom_gates.DDGate,
    "parallel": qss.custom_gates.ParallelGates,
    "stripped_cz": qss.custom_gates.StrippedCZGate,
    "zzswap": qss.custom_gates.ZZSwapGate,
    "ix": qss.custom_gates.iXGate,
    "ixdg": qss.custom_gates.iXdgGate,
    "iccx": qss.custom_gates.iCCXGate,
    "iccxdg": qss.custom_gates.iCCXdgGate,
    "ms": qiskit.circuit.library.MSGate,
}

# Custom resolvers, necessary when `gate != type(gate)(*gate.params)`
# MSGate included as a workaround for https://github.com/Qiskit/qiskit/issues/11378
_custom_resolvers: dict[
    type[qiskit.circuit.Instruction],
    Callable[[qiskit.circuit.Instruction], qiskit.circuit.Instruction],
] = {
    qss.custom_gates.ParallelGates: lambda gate: qss.custom_gates.ParallelGates(
        *[_resolve_gate(inst.operation) for inst in gate.definition], label=gate.label
    ),
    qiskit.circuit.library.MSGate: lambda gate: qiskit.circuit.library.MSGate(
        gate.num_qubits, gate.params[0], label=gate.label
    ),
}


def _mcphase(
    lam: float | qiskit.circuit.ParameterExpression,
    num_ctrl_qubits: int,
    label: str | None = None,
    ctrl_state: str | int | None = None,
) -> qiskit.circuit.library.MCPhaseGate:
    """The `ctrl_state` argument was added to `MCPhaseGate` in Qiskit 1.1.0."""

    if qiskit.__version__.split(".")[:2] >= ["1", "1"]:
        return qiskit.circuit.library.MCPhaseGate(
            lam, num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state
        )

    gate = qiskit.circuit.library.MCPhaseGate(lam, num_ctrl_qubits=num_ctrl_qubits, label=label)
    gate.ctrl_state = ctrl_state
    return gate


# Resolvers for controlled gates which can be incorrectly serialized by Qiskit
_controlled_gate_resolvers: dict[
    type[qiskit.circuit.Instruction],
    Callable[[qiskit.circuit.Instruction], qiskit.circuit.Instruction],
] = {
    qiskit.circuit.library.PhaseGate: lambda gate: _mcphase(
        gate.params[0], gate.num_ctrl_qubits, ctrl_state=gate.ctrl_state, label=gate.label
    ),
    qiskit.circuit.library.U1Gate: lambda gate: qiskit.circuit.library.MCU1Gate(
        gate.params[0], gate.num_ctrl_qubits, ctrl_state=gate.ctrl_state, label=gate.label
    ),
    qiskit.circuit.library.XGate: lambda gate: qiskit.circuit.library.MCXGate(
        gate.num_ctrl_qubits, ctrl_state=gate.ctrl_state, label=gate.label
    ),
}


if hasattr(qiskit.circuit.library, "QFTGate"):
    # QFTGate introduced in qiskit 1.2.0

    QFTGate = getattr(qiskit.circuit.library, "QFTGate")

    _controlled_gate_resolvers[QFTGate] = lambda gate: QFTGate(
        gate.num_qubits - gate.num_ctrl_qubits
    ).control(
        num_ctrl_qubits=gate.num_ctrl_qubits,
        ctrl_state=gate.ctrl_state,
        label=gate.label,
    )


def json_encoder(val: object) -> dict[str, object]:
    """Converts (real or complex) arrays to a JSON-serializable format.

    Args:
        val: The value to be serialized.

    Returns:
        A JSON dictionary containing the provided name and array values.

    Raises:
        TypeError: If `val` is not a `np.ndarray`.
    """
    if isinstance(val, np.ndarray):
        return {
            "type": "qss_array",
            "real": val.real.tolist(),
            "imag": val.imag.tolist(),
        }

    raise TypeError(f"Object of type {type(val)} is not JSON serializable.")


def json_resolver(val: T) -> T | npt.NDArray[np.complex128]:
    """Hook to deserialize objects that were serialized via `json_encoder()`.

    Args:
        val: The deserialized object.

    Returns:
        The resolved object.
    """
    if isinstance(val, dict) and val.get("type") == "qss_array":
        real_part = val.get("real", 0)
        imag_part = val.get("imag", 0)
        return np.array(real_part) + 1j * np.array(imag_part)

    return val


def to_json(val: object) -> str:
    """Extends `json.dumps` to support numpy arrays.

    Args:
        val: The value to be serialized.

    Returns:
        The JSON-serialized value (a string).
    """
    return json.dumps(val, default=json_encoder)


def serialize_circuits(circuits: qiskit.QuantumCircuit | Sequence[qiskit.QuantumCircuit]) -> str:
    """Serializes qiskit.QuantumCircuit(s) into a single string.

    Args:
        circuits: A `qiskit.QuantumCircuit` or list of `qiskit.QuantumCircuit` to be serialized.

    Returns:
        A string representing the serialized circuit(s).
    """
    if isinstance(circuits, qiskit.QuantumCircuit):
        circuits = [_prepare_circuit(circuits)]
    else:
        circuits = [_prepare_circuit(circuit) for circuit in circuits]

    # Use the lowest compatible QPY version for serialization. Deserialization can't be done with a
    # QPY version older than that used for serialization, so this prevents us from having to force
    # users to update Qiskit the moment we do
    buf = io.BytesIO()
    qiskit.qpy.dump(circuits, buf, version=QPY_SERIALIZATION_VERSION)
    return gss.serialization.bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> list[qiskit.QuantumCircuit]:
    """Deserializes serialized qiskit.QuantumCircuit(s).

    Args:
        serialized_circuits: String generated via `qss.serialization.serialize_circuit()`.

    Returns:
        A list containing the deserialized circuits.

    Raises:
        ValueError: If `serialized_circuits` can't be deserialized.
    """
    buf = io.BytesIO(gss.serialization.str_to_bytes(serialized_circuits))

    try:
        with warnings.catch_warnings(record=False):
            warnings.filterwarnings("ignore", "The qiskit version", UserWarning, "qiskit")
            circuits = qiskit.qpy.load(buf)

    except Exception as e:
        qpy_version_match = re.match(b"QISKIT([\x00-\xff])", buf.getvalue())
        circuits_qpy_version = ord(qpy_version_match.group(1)) if qpy_version_match else 0
        if circuits_qpy_version > qiskit.qpy.common.QPY_VERSION:
            # If the circuit was serialized with a newer version of QPY, that's probably what caused
            # this error. In this case we should just tell the user to update.
            raise ValueError(
                "Circuits failed to deserialize. This is likely because your version of Qiskit "
                f"({qiskit.__version__}) is out of date. Consider updating it."
            )
        else:
            # Otherwise there is probably a more complicated issue.
            raise ValueError(
                "Circuits failed to deserialize. Please contact superstaq@infleqtion.com or file a "
                "report at https://github.com/Infleqtion/client-superstaq/issues containing "
                "the following information (as well as any other relevant context):\n\n"
                f"qiskit-superstaq version: {qss.__version__}\n"
                f"qiskit version: {qiskit.__version__}\n"
                f"error: {e!r}"
            )

    return [_resolve_circuit(circuit) for circuit in circuits]


def insert_times_and_durations(
    circuit: qiskit.QuantumCircuit,
    durations: Sequence[int],
    start_times: Sequence[int],
) -> qiskit.QuantumCircuit:
    """Adds timing info to a circuit.

    This is a workaround for https://github.com/Qiskit/qiskit/issues/11879.

    Args:
        circuit: The circuit to add timing information to.
        durations: A list containing the duration of every instruction in `circuit`.
        start_times: A list containing the start_time of every instruction in `circuit`.

    Returns:
        A new circuit, in which the `.duration` attribute of every gate has been filled-in, as well
        as the `.duration` `.op_start_times` attributes of the circuit itself.
    """
    new_circuit = circuit.copy_empty_like()
    circuit_duration = 0
    for inst, duration, start_time in zip(circuit, durations, start_times):
        operation = inst.operation
        if inst.operation.duration != duration:
            operation = inst.operation.to_mutable()
            operation.duration = duration
            inst = inst.replace(operation=operation)
        circuit_duration = max(circuit_duration, start_time + duration)
        new_circuit.append(inst)

    if len(new_circuit) == len(circuit):
        new_circuit._op_start_times = start_times
        new_circuit.duration = circuit_duration
        return new_circuit

    return circuit


def _is_qiskit_gate(gate: qiskit.circuit.Instruction) -> bool:
    """Returns True if `gate` will be correctly resolved by QPY."""
    base_class = getattr(gate, "base_class", type(gate))

    return (
        issubclass(base_class, qiskit.circuit.Instruction)
        and base_class.__module__.startswith("qiskit.")
        and base_class
        not in (
            qiskit.circuit.Instruction,
            qiskit.circuit.Gate,
            qiskit.circuit.ControlledGate,
        )
        and not issubclass(  # https://github.com/Qiskit/qiskit/issues/11378
            base_class, qiskit.circuit.library.MSGate
        )
        and not (
            issubclass(  # https://github.com/Qiskit/qiskit/issues/11377
                base_class,
                (
                    qiskit.circuit.library.MCXGate,
                    qiskit.circuit.library.MCU1Gate,
                    qiskit.circuit.library.MCPhaseGate,
                ),
            )
            and gate.ctrl_state != 2**gate.num_ctrl_qubits - 1
        )
        and (
            hasattr(qiskit.circuit.library, base_class.__name__)
            or hasattr(qiskit.circuit, base_class.__name__)
            or hasattr(qiskit.circuit.controlflow, base_class.__name__)
        )
    )


def _prepare_circuit(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Rewrites the input circuit in anticipation of QPY serialization.

    This is intended to be run prior to serialization as a workaround for various known QPY issues,
    namely:
    * https://github.com/Qiskit/qiskit/issues/11378 (mishandling of `MSGate`)
    * https://github.com/Qiskit/qiskit/issues/11377 (mishandling of multi-controlled gates)
    * https://github.com/Qiskit/qiskit/issues/8941 (incorrect definitions for gates sharing a name)
    * https://github.com/Qiskit/qiskit/issues/8794 (serialization error for non-default ctrl_state)
    * https://github.com/Qiskit/qiskit/issues/8549 (incorrect gate names in deserialized circuit)

    Most significantly (#8941 above), QPY requires unique custom gates to have unique `.name`
    attributes (including parameterized gates differing by just their `.params` attributes). This
    routine ensures this by wrapping unequal gates with the same name into uniquely-named temporary
    instructions. The original circuit can then be recovered using the `_resolve_circuit` function
    below.

    Args:
        circuit: The `qiskit.QuantumCircuit` to be rewritten.

    Returns:
        A copy of the input circuit with unique custom instruction names.
    """
    old_gates_by_name = {}
    new_gates_by_name = {}

    def _update_gate(gate: qiskit.circuit.Instruction) -> qiskit.circuit.Instruction:
        # Control flow operations contain nested circuit blocks; prepare them first
        if isinstance(gate, qiskit.circuit.ControlFlowOp):
            gate = gate.replace_blocks([_prepare_circuit(block) for block in gate.blocks])

        # Check if this is a gate QPY already handles correctly
        if _is_qiskit_gate(gate):
            return gate

        if gate.name not in old_gates_by_name:
            new_gate = _prepare_gate(gate)
            old_gates_by_name[gate.name] = [gate]
            new_gates_by_name[gate.name] = [new_gate]
            return new_gate

        for i, other in enumerate(old_gates_by_name[gate.name]):
            if gate is other or gate == other:
                return new_gates_by_name[gate.name][i]

        # Workaround for https://github.com/Qiskit/qiskit/issues/8941: wrap gate in a temporary
        # instruction to prevent `.definition` from being overwritten
        new_gate = _wrap_gate(gate)
        old_gates_by_name[gate.name].append(gate)
        new_gates_by_name[gate.name].append(new_gate)
        return new_gate

    new_circuit = circuit.copy_empty_like()
    for inst in circuit:
        new_inst = inst.replace(operation=_update_gate(inst.operation))
        new_circuit.append(new_inst)

    return new_circuit


def _prepare_gate(gate: qiskit.circuit.Instruction) -> qiskit.circuit.Instruction:
    # Check if this is a gate QPY already handles
    if _is_qiskit_gate(gate):
        return gate

    # Workaround for https://github.com/Qiskit/qiskit/issues/8794
    if isinstance(gate, qiskit.circuit.ControlledGate):
        if gate.definition is not None and not gate._definition:
            gate._define()

    if isinstance(gate, tuple(_custom_gates_by_name.values())) and not issubclass(
        gate.base_class, tuple(_custom_resolvers)
    ):
        return gate

    if isinstance(gate, qiskit.circuit.ControlledGate):
        return qiskit.circuit.ControlledGate(
            name=gate._name,
            num_qubits=gate.num_qubits,
            params=gate.params,
            label=gate.label,
            num_ctrl_qubits=gate.num_ctrl_qubits,
            definition=_prepare_circuit(gate._definition),
            ctrl_state=gate.ctrl_state,
            base_gate=_prepare_gate(gate.base_gate),
        )

    if isinstance(gate, qiskit.circuit.Gate):
        new_gate = qiskit.circuit.Gate(
            gate.name,
            num_qubits=gate.num_qubits,
            params=gate.params,
            label=gate.label,
        )

    else:
        new_gate = qiskit.circuit.Instruction(
            gate.name,
            num_qubits=gate.num_qubits,
            num_clbits=gate.num_clbits,
            params=gate.params,
            label=gate.label,
        )

    if gate.definition:
        new_gate.definition = _prepare_circuit(gate.definition)

    return new_gate


def _wrap_gate(gate: qiskit.circuit.Instruction) -> qiskit.circuit.Instruction:
    """Wrap `gate` in a uniquely=name instruction containing only that gate in its `.definition`.

    This functions as a workaround for https://github.com/Qiskit/qiskit/issues/8941.
    """

    name = f"__superstaq_wrapper_{id(gate)}"
    circuit = qiskit.QuantumCircuit(gate.num_qubits, gate.num_clbits, name=name)
    circuit.append(_prepare_gate(gate), range(gate.num_qubits), range(gate.num_clbits))
    new_gate = circuit.to_instruction(label=gate.name)

    # For backwards compatibility
    compat_name = "parallel_gates" if isinstance(gate, qss.ParallelGates) else gate.name
    new_gate.definition.name = compat_name
    new_gate.params.extend(gate.params)

    return new_gate


def _resolve_circuit(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Reverse of the transformation performed by `_prepare_circuit`.

    This is intended to be run after deserialization in order to recover the names and types of
    instructions in the original circuit.
    """
    new_circuit = circuit.copy_empty_like()
    for inst in circuit:
        inst = inst.replace(operation=_resolve_gate(inst.operation))
        new_circuit.append(inst)
    return new_circuit


def _resolve_gate(gate: qiskit.circuit.Instruction) -> qiskit.circuit.Instruction:
    if gate.name.startswith(r"__superstaq_wrapper_"):
        return _resolve_gate(gate.definition[0].operation)

    if isinstance(gate, qiskit.circuit.ControlFlowOp):
        return gate.replace_blocks([_resolve_circuit(block) for block in gate.blocks])

    if type(gate) is qiskit.circuit.ControlledGate:
        gate.base_gate = _resolve_gate(gate.base_gate)

        if gate.definition and gate._definition:
            gate.definition = _resolve_circuit(gate._definition)

        if resolver := _controlled_gate_resolvers.get(gate.base_gate.base_class):
            trial_gate = resolver(gate)

            if trial_gate.definition and not trial_gate._definition:
                trial_gate._define()

            if (
                trial_gate.definition == gate.definition
                or trial_gate._definition == gate._definition
            ):
                return trial_gate

    elif not gate.mutable or type(gate) not in (qiskit.circuit.Instruction, qiskit.circuit.Gate):
        return gate

    elif gate.definition is not None:
        gate.definition = _resolve_circuit(gate.definition)

    return _resolve_custom_gate(gate)


def _resolve_custom_gate(gate: qiskit.circuit.Instruction) -> qiskit.circuit.Instruction:
    for name, trial_class in _custom_gates_by_name.items():
        if gate.name == name or gate.name.startswith(f"{name}_"):
            try:
                if resolver := _custom_resolvers.get(trial_class):
                    trial_gate = resolver(gate)
                elif issubclass(trial_class, qiskit.circuit.ControlledGate):
                    trial_gate = trial_class(
                        *gate.params, ctrl_state=gate.ctrl_state, label=gate.label
                    )
                else:
                    trial_gate = trial_class(*gate.params, label=gate.label)

            except Exception:
                continue

            if trial_gate.definition == gate.definition:
                return trial_gate

    return gate
