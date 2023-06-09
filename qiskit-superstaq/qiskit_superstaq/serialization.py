import io
import re
import warnings
from typing import Dict, List, Set, Tuple, Union

import general_superstaq as gss
import qiskit
import qiskit.qpy
from qiskit.converters.ast_to_dag import AstInterpreter

import qiskit_superstaq as qss


def _assign_unique_inst_names(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """QPY requires unique custom gates to have unique `.name` attributes (including parameterized
    gates differing by just their `.params` attributes). This function rewrites the input circuit
    with new instruction names given by appending a unique (consecutive) "_{index}" string to the
    name of any custom instruction which shares a name with a non-equivalent prior instruction in
    the circuit.

    Args:
        circuit: qiskit.QuantumCircuit to be rewritten

    Returns:
        A copy of the input circuit with unique custom instruction names
    """

    unique_insts_by_name: Dict[str, List[qiskit.circuit.Instruction]] = {}
    insts_to_update: List[Tuple[int, int]] = []
    unique_inst_ids: Set[int] = set()

    qiskit_gates = set(AstInterpreter.standard_extension) | {"measure"}

    new_circuit = circuit.copy()
    for inst, _, _ in new_circuit:
        inst._define()
        if inst.name in qiskit_gates or id(inst) in unique_inst_ids:
            continue

        # save id() in case instruction instance is used more than once
        unique_inst_ids.add(id(inst))

        if inst.name in unique_insts_by_name:
            index = 0
            for other in unique_insts_by_name[inst.name]:
                if inst == other:
                    break
                index += 1

            if index == len(unique_insts_by_name[inst.name]):
                unique_insts_by_name[inst.name].append(inst)
            if index > 0:
                insts_to_update.append((inst, index))
        else:
            unique_insts_by_name[inst.name] = [inst]

    for inst, index in insts_to_update:
        inst.name += f"_{index}"

    return new_circuit


def serialize_circuits(circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]) -> str:
    """Serialize QuantumCircuit(s) into a single string

    Args:
        circuits: a QuantumCircuit or list of QuantumCircuits to be serialized

    Returns:
        str representing the serialized circuit(s)
    """
    if isinstance(circuits, qiskit.QuantumCircuit):
        circuits = [_assign_unique_inst_names(circuits)]
    else:
        circuits = [_assign_unique_inst_names(circuit) for circuit in circuits]

    buf = io.BytesIO()
    qiskit.qpy.dump(circuits, buf)
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
        qpy_version_match = re.match(b"QISKIT(.)", buf.getvalue())
        circuits_qpy_version = ord(qpy_version_match.group(1)) if qpy_version_match else 0
        if circuits_qpy_version > qiskit.qpy.common.QPY_VERSION:
            # If the circuit was serialized with a newer version of QPY, that's probably what caused
            # this error. In this case we should just tell the user to update.
            raise ValueError(
                "Circuits failed to deserialize. This is likely because your version of "
                f"qiskit-terra ({qiskit.__version__}) is out of date. Consider updating it."
            )
        else:
            # Otherwise there is probably a more complicated issue.
            raise ValueError(
                "Circuits failed to deserialize. Please contact info@super.tech or file a "
                "report at https://github.com/SupertechLabs/client-superstaq/issues containing "
                "the following information (as well as any other relevant context):\n\n"
                f"qiskit-superstaq version: {qss.__version__}\n"
                f"qiskit-terra version: {qiskit.__version__}\n"
                f"error: {e!r}"
            )

    for circuit in circuits:
        for pc, (inst, qargs, cargs) in enumerate(circuit._data):
            new_inst = qss.custom_gates.custom_resolver(inst)
            if new_inst is not None:
                circuit._data[pc] = qiskit.circuit.CircuitInstruction(new_inst, qargs, cargs)

    return circuits
