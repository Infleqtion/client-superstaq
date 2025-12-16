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

import codecs
import pickle
from typing import Any


def bytes_to_str(bytes_data: bytes) -> str:
    """Convert arbitrary bytes data into a string.

    Args:
        bytes_data: The data to be converted.

    Returns:
        The string from conversion.
    """
    return codecs.encode(bytes_data, "base64").decode()


def str_to_bytes(str_data: str) -> bytes:
    """Decode the string-encoded bytes data returned by `bytes_to_str`.

    Args:
        str_data: The string data to be decoded.

    Returns:
        The decoded by bytes data.
    """
    return codecs.decode(str_data.encode(), "base64")


def serialize(obj: Any) -> str:
    """Serialize picklable object into a string.

    Args:
        obj: A picklable object to be serialized.

    Returns:
        The string representing the serialized object.
    """
    return bytes_to_str(pickle.dumps(obj))


def deserialize(serialized_obj: str) -> Any:
    """Deserialize serialized objects.

    Args:
        serialized_obj: A string generated via `general_superstaq.serialization.serialize`.

    Returns:
        The serialized object.
    """
    return pickle.loads(str_to_bytes(serialized_obj))
