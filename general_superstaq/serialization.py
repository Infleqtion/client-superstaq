import codecs
import pickle
from typing import Any


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
        serialized_obj: a str generated via general_superstaq.serialization.serialize()

    Returns:
        the serialized object
    """

    return pickle.loads(_str_to_bytes(serialized_obj))
