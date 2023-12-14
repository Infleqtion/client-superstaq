from __future__ import annotations

import io
import json
import re
import warnings
from typing import Callable, Dict, List, Sequence, TypeVar, Union

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit.qpy
import qiskit_ibm_provider

import qiskit_superstaq as qss

T = TypeVar("T")
RealArray = Union[int, float, List["RealArray"]]

# Custom gate types to resolve when deserializing circuits
# MSGate included as a workaround for https://github.com/Qiskit/qiskit/issues/11378
_custom_gates_by_name: Dict[str, type[qiskit.circuit.Instruction]] = {
    "acecr": qss.custom_gates.AceCR,
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
_custom_resolvers: Dict[
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


def json_encoder(val: object) -> Dict[str, Union[str, RealArray]]:
    """Convert (real or complex) arrays to a JSON-serializable format.

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


def json_resolver(val: T) -> Union[T, npt.NDArray[np.complex_]]:
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


def serialize_circuits(
    circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]]
) -> str:
    """Serialize QuantumCircuit(s) into a single string.

    Args:
        circuits: A `qiskit.QuantumCircuit` or list of `qiskit.QuantumCircuit`s to be serialized.

    Returns:
        A string representing the serialized circuit(s).
    """
    if isinstance(circuits, qiskit.QuantumCircuit):
        circuits = [_prepare_circuit(circuits)]
    else:
        circuits = [_prepare_circuit(circuit) for circuit in circuits]

    # Use `qiskit_ibm_provider.qpy` for serialization, which is a delayed copy of `qiskit.qpy`.
    # Deserialization can't be done with a QPY version older than that used for serialization, so
    # this prevents us from having to force users to update Qiskit the moment we do (this is what
    # Qiskit itself does for circuit submission)
    buf = io.BytesIO()
    qiskit_ibm_provider.qpy.dump(circuits, buf)
    return gss.serialization.bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    """Deserialize serialized `qiskit.QuantumCircuit`(s).

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
        and not issubclass(  # Qiskit mishandles these gates
            base_class,
            (
                qiskit.circuit.library.MSGate,  # https://github.com/Qiskit/qiskit/issues/11378
                qiskit.circuit.library.MCXGate,  # https://github.com/Qiskit/qiskit/issues/11377
                qiskit.circuit.library.MCU1Gate,
                qiskit.circuit.library.MCPhaseGate,
            ),
        )
        and (
            hasattr(qiskit.circuit.library, base_class.__name__)
            or hasattr(qiskit.circuit, base_class.__name__)
            or hasattr(qiskit.extensions, base_class.__name__)
            or hasattr(qiskit.extensions.quantum_initializer, base_class.__name__)
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


def _prepare_gate(
    gate: qiskit.circuit.Instruction, force_wrapper: bool = False
) -> qiskit.circuit.Instruction:
    # Check if this is a gate QPY already handles
    if _is_qiskit_gate(gate):
        return gate

    # Workaround for https://github.com/Qiskit/qiskit/issues/8794
    if isinstance(gate, qiskit.circuit.ControlledGate):
        if gate.definition is not None and not gate._definition:
            gate._define()

    if isinstance(gate, tuple(_custom_gates_by_name.values())) and not isinstance(
        gate, tuple(_custom_resolvers)
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

    elif isinstance(gate, qiskit.circuit.ControlFlowOp):
        return gate.replace_blocks([_resolve_circuit(block) for block in gate.blocks])

    elif type(gate) is qiskit.circuit.ControlledGate:
        gate.base_gate = _resolve_gate(gate.base_gate)
        gate.name = gate._name.rsplit("_", 1)[0]  # https://github.com/Qiskit/qiskit/issues/8549

        if gate.definition is not None and gate._definition is not None:
            gate.definition = _resolve_circuit(gate._definition)

    elif type(gate) in (qiskit.circuit.Instruction, qiskit.circuit.Gate):
        if gate.definition:
            gate.definition = _resolve_circuit(gate.definition)

    else:
        return gate

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
