import io
from typing import List, Union

import applications_superstaq
import qiskit
import qiskit.circuit.qpy_serialization


def serialize_circuits(circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]) -> str:
    """Serialize QuantumCircuit(s) into a single string

    Args:
        circuits: a QuantumCircuit or list of QuantumCircuits to be serialized

    Returns:
        str representing the serialized circuit(s)
    """
    buf = io.BytesIO()
    qiskit.circuit.qpy_serialization.dump(circuits, buf)
    return applications_superstaq.converters._bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    """Deserialize serialized QuantumCircuit(s)

    Args:
        serialized_circuits: str generated via qiskit_superstaq.serialization.serialize_circuit()

    Returns:
        a list of QuantumCircuits
    """
    buf = io.BytesIO(applications_superstaq.converters._str_to_bytes(serialized_circuits))
    return qiskit.circuit.qpy_serialization.load(buf)
