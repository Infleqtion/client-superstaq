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
    buf = io.BytesIO()
    qiskit.circuit.qpy_serialization.dump(circuits, buf)
    return bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    buf = io.BytesIO(str_to_bytes(serialized_circuits))
    return qiskit.circuit.qpy_serialization.load(buf)
