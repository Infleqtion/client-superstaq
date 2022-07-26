import io
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
        if inst.name in qiskit_gates or id(inst) in unique_inst_ids:
            continue

        # save id() in case instruction instance is used more than once
        unique_inst_ids.add(id(inst))

        if inst.name in unique_insts_by_name:
            index = 0
            for other in unique_insts_by_name[inst.name]:
                # compare qasm strings first because equality checking is very slow
                if inst.qasm() == other.qasm() and inst == other:
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
    return gss.converters._bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    """Deserialize serialized QuantumCircuit(s)

    Args:
        serialized_circuits: str generated via qss.serialization.serialize_circuit()

    Returns:
        a list of QuantumCircuits
    """
    buf = io.BytesIO(gss.converters._str_to_bytes(serialized_circuits))

    with warnings.catch_warnings(record=False):
        warnings.filterwarnings("ignore", "The qiskit version", UserWarning, "qiskit")
        circuits = qiskit.qpy.load(buf)

    for circuit in circuits:
        for pc, (inst, qargs, cargs) in enumerate(circuit._data):
            new_inst = qss.custom_gates.custom_resolver(inst)
            if new_inst is not None:
                circuit._data[pc] = (new_inst, qargs, cargs)

    return circuits
