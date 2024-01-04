from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Sequence

import cirq

import cirq_superstaq as css

SUPERSTAQ_RESOLVERS = [
    css.ops.qudit_gates.custom_resolver,
    css.ops.qubit_gates.custom_resolver,
]


def serialize_circuits(circuits: cirq.AbstractCircuit | Sequence[cirq.AbstractCircuit]) -> str:
    """Serialize circuit(s) into a json string.

    Args:
        circuits: A `cirq.Circuit` or list of `cirq.Circuits` to be serialized.

    Returns:
        A string representing the serialized circuit(s).
    """
    return cirq.to_json(circuits)


def deserialize_circuits(serialized_circuits: str) -> list[cirq.Circuit]:
    """Deserialize serialized circuit(s).

    Args:
        serialized_circuits: A json string generated via `serialization.serialize_circuit()`.

    Returns:
        The circuit or list of circuits that was serialized.
    """
    resolvers = [*SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS]
    circuits = cirq.read_json(json_text=serialized_circuits, resolvers=resolvers)
    if isinstance(circuits, cirq.Circuit):
        return [circuits]
    return circuits


def deserialize_qiskit_circuits(
    serialized_qiskit_circuits: str, circuits_is_list: bool
) -> list[object] | None:
    """Deserializes `qiskit.QuantumCircuit` objects, if possible; otherwise warns the user.

    Args:
        serialized_qiskit_circuits: Qiskit circuits serialized via `qss.serialize_circuits()`.
        circuits_is_list: Whether to refer to "circuits" (plural) or "circuit" (singular) in warning
            messages.

    Returns:
        A list of deserialized `qiskit.QuantumCircuit` objects, or None if the provided circuits
        could not be deserialized.
    """
    if importlib.util.find_spec("qiskit_superstaq"):
        import qiskit
        import qiskit_superstaq as qss

        try:
            return qss.deserialize_circuits(serialized_qiskit_circuits)
        except Exception as e:
            s = "s" if circuits_is_list else ""
            warnings.warn(
                f"Your compiled pulse gate circuit{s} could not be deserialized. Please "
                "make sure your qiskit-superstaq installation is up-to-date (by running "
                "`pip install -U qiskit-superstaq`).\n\n"
                "If the problem persists, please let us know at superstaq@infleqtion.com, "
                "or file a report at https://github.com/Infleqtion/client-superstaq/issues "
                "containing the following information (and any other relevant context):\n\n"
                f"cirq-superstaq version: {css.__version__}\n"
                f"qiskit-superstaq version: {qss.__version__}\n"
                f"qiskit version: {qiskit.__version__}\n"
                f"error: {e!r}\n\n"
                f"You can still access your compiled circuit{s} using the .circuit{s} "
                "attribute of this output."
            )

    else:
        s = "s" if circuits_is_list else ""
        warnings.warn(
            "qiskit-superstaq is required to deserialize compiled pulse gate circuits. You can "
            "install it with `pip install qiskit-superstaq`.\n\n"
            f"You can still access your compiled circuit{s} using the .circuit{s} attribute of "
            "this output."
        )

    return None
