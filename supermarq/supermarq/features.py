import cirq

import supermarq as sm


def compute_communication(  # pylint: disable=missing-function-docstring
    circuit: cirq.Circuit,
) -> float:
    return sm.converters.compute_communication_with_qiskit(sm.converters.cirq_to_qiskit(circuit))


def compute_liveness(circuit: cirq.Circuit) -> float:  # pylint: disable=missing-function-docstring
    return sm.converters.compute_liveness_with_qiskit((sm.converters.cirq_to_qiskit(circuit)))


def compute_parallelism(  # pylint: disable=missing-function-docstring
    circuit: cirq.Circuit,
) -> float:
    return sm.converters.compute_parallelism_with_qiskit(sm.converters.cirq_to_qiskit(circuit))


def compute_measurement(  # pylint: disable=missing-function-docstring
    circuit: cirq.Circuit,
) -> float:
    return sm.converters.compute_measurement_with_qiskit(sm.converters.cirq_to_qiskit(circuit))


def compute_entanglement(  # pylint: disable=missing-function-docstring
    circuit: cirq.Circuit,
) -> float:
    return sm.converters.compute_entanglement_with_qiskit(sm.converters.cirq_to_qiskit(circuit))


def compute_depth(circuit: cirq.Circuit) -> float:  # pylint: disable=missing-function-docstring
    return sm.converters.compute_depth_with_qiskit(sm.converters.cirq_to_qiskit(circuit))
