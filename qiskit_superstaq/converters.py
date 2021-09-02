import codecs
import io
from typing import List, Union

import qiskit
import qiskit.circuit.qpy_serialization


def bytes_to_str(bytes_data: bytes) -> str:
    return codecs.encode(bytes_data, "base64").decode()


def str_to_bytes(str_data: str) -> bytes:
    return codecs.decode(str_data.encode(), "base64")


def serialize_circuits(circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]) -> str:
    """Serialize QuantumCircuits into a single string

    Args:
        circuits: a QuantumCircuit or list of QuantumCircuits to be serialized

    Returns:
        str representing the serialized circuits
    """
    buf = io.BytesIO()
    qiskit.circuit.qpy_serialization.dump(circuits, buf)
    return bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    """Deserialize serialized QuantumCircuit(s)

    Args:
        serialized_circuits: str generated via qiskit_superstaq.converters.serialize_circuit()

    Returns:
        a list of QuantumCircuits
    """
    buf = io.BytesIO(str_to_bytes(serialized_circuits))
    return qiskit.circuit.qpy_serialization.load(buf)
