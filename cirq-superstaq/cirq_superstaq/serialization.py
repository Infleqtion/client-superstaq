from typing import List, Sequence, Union

import cirq

import cirq_superstaq as css

SUPERSTAQ_RESOLVERS = [
    css.ops.qudit_gates.custom_resolver,
    css.ops.qubit_gates.custom_resolver,
]


def serialize_circuits(
    circuits: Union[cirq.AbstractCircuit, Sequence[cirq.AbstractCircuit]]
) -> str:
    """Serialize circuit(s) into a json string.

    Args:
        circuits: A `cirq.Circuit` or list of `cirq.Circuits` to be serialized.

    Returns:
        A string representing the serialized circuit(s).
    """
    return cirq.to_json(circuits)


def deserialize_circuits(serialized_circuits: str) -> List[cirq.Circuit]:
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
