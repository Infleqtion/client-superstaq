import codecs
import io
import pickle
from typing import Any, List, Union

import qiskit
import qiskit.circuit.qpy_serialization


def _bytes_to_str(bytes_data: bytes) -> str:
    return codecs.encode(bytes_data, "base64").decode()


def _str_to_bytes(str_data: str) -> bytes:
    return codecs.decode(str_data.encode(), "base64")


def serialize(obj: Any) -> str:
    """Serialize picklable object into a string

    Args:
        obj: a picklable object to be serialized

    Returns:
        str representing the serialized object
    """

    return _bytes_to_str(pickle.dumps(obj))


def deserialize(serialized_obj: str) -> Any:
    """Deserialize serialized objects

    Args:
        serialized_obj: a str generated via qiskit_superstaq.converters.serialize()

    Returns:
        the serialized object
    """

    return pickle.loads(_str_to_bytes(serialized_obj))


def serialize_circuits(circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]) -> str:
    """Serialize QuantumCircuits into a single string

    Args:
        circuits: a QuantumCircuit or list of QuantumCircuits to be serialized

    Returns:
        str representing the serialized circuits
    """
    buf = io.BytesIO()
    qiskit.circuit.qpy_serialization.dump(circuits, buf)
    return _bytes_to_str(buf.getvalue())


def deserialize_circuits(serialized_circuits: str) -> List[qiskit.QuantumCircuit]:
    """Deserialize serialized QuantumCircuit(s)

    Args:
        serialized_circuits: str generated via qiskit_superstaq.converters.serialize_circuit()

    Returns:
        a list of QuantumCircuits
    """
    buf = io.BytesIO(_str_to_bytes(serialized_circuits))
    return qiskit.circuit.qpy_serialization.load(buf)
