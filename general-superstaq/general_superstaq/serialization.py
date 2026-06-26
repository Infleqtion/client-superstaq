# Copyright 2026 Infleqtion
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
import importlib.util
import pickle
import warnings
from collections.abc import Sequence
from typing import Any

import general_superstaq as gss


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
    return pickle.loads(str_to_bytes(serialized_obj))  # noqa: S301


def deserialize_qiskit_circuits(  # pragma: no cover (requires `qiskit_superstaq` install)
    serialized_qiskit_circuits: str,
    circuits_is_list: bool,
    pulse_start_times: Sequence[Sequence[int]] | None = None,
) -> list[object] | None:
    """Deserializes `qiskit.QuantumCircuit` objects, if possible; otherwise warns the user.

    Args:
        serialized_qiskit_circuits: Qiskit circuits serialized via `qss.serialize_circuits()`.
        circuits_is_list: Whether to refer to "circuits" (plural) or "circuit" (singular) in warning
            messages.
        pulse_start_times: A list of lists of start times, where each list contains the start times
            of every op in the corresponding (serialized) circuit.

    Returns:
        A list of deserialized `qiskit.QuantumCircuit` objects, or None if the provided circuits
        could not be deserialized.
    """
    if importlib.util.find_spec("qiskit_superstaq"):
        import qiskit  # noqa: PLC0415
        import qiskit_superstaq as qss  # noqa: PLC0415

        try:
            pulse_gate_circuits = qss.deserialize_circuits(serialized_qiskit_circuits)

        except Exception as e:
            s = "s" if circuits_is_list else ""
            warnings.warn(
                f"Your compiled pulse gate circuit{s} could not be deserialized. Please "
                "make sure your qiskit-superstaq installation is up-to-date (by running "
                "`pip install -U qiskit-superstaq`).\n\n"
                "If the problem persists, please let us know at superstaq@infleqtion.com, "
                "or file a report at https://github.com/Infleqtion/client-superstaq/issues "
                "containing the following information (and any other relevant context):\n\n"
                f"general-superstaq version: {gss.__version__}\n"
                f"qiskit-superstaq version: {qss.__version__}\n"
                f"qiskit version: {qiskit.__version__}\n"
                f"error: {e!r}\n\n"
                f"You can still access your compiled circuit{s} using the .circuit{s} "
                "attribute of this output.",
                stacklevel=2,
            )
        else:
            if pulse_start_times:
                for circuit, start_times in zip(pulse_gate_circuits, pulse_start_times):
                    circuit._op_start_times = start_times

            return pulse_gate_circuits

    else:
        s = "s" if circuits_is_list else ""
        warnings.warn(
            "qiskit-superstaq is required to deserialize compiled pulse gate circuits. You can "
            "install it with `pip install qiskit-superstaq`.\n\n"
            f"You can still access your compiled circuit{s} using the .circuit{s} attribute of "
            "this output.",
            stacklevel=2,
        )

    return None
