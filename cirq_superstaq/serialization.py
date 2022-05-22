import json
from typing import List, Sequence, Union

import cirq

import cirq_superstaq as css


def serialize_circuits(
    circuits: Union[cirq.AbstractCircuit, Sequence[cirq.AbstractCircuit]]
) -> str:
    """Serialize Circuit(s) into a json string

    Args:
        circuits: a Circuit or list of Circuits to be serialized

    Returns:
        str representing the serialized circuit(s)
    """
    dt = json.loads(cirq.to_json(circuits))
    if isinstance(dt, list):
        for circuit_dt in dt:
            if "device" not in circuit_dt:
                circuit_dt["device"] = {"cirq_type": "_UnconstrainedDevice"}
    else:
        if "device" not in dt:
            dt["device"] = {"cirq_type": "_UnconstrainedDevice"}

    return json.dumps(dt)


def deserialize_circuits(serialized_circuits: str) -> List[cirq.Circuit]:
    """Deserialize serialized Circuit(s)

    Args:
        serialized_circuits: json str generated via converters.serialize_circuit()

    Returns:
        the Circuit or list of Circuits that was serialized
    """
    resolvers = [css.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    circuits = cirq.read_json(json_text=serialized_circuits, resolvers=resolvers)
    if isinstance(circuits, cirq.Circuit):
        return [circuits]
    return circuits
