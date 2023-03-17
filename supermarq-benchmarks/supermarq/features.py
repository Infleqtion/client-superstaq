import cirq

import supermarq


def compute_communication(circuit: cirq.Circuit) -> float:
    """Compute the *communication* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the communication feature for this circuit.
    """
    return supermarq.converters.compute_communication_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )


def compute_liveness(circuit: cirq.Circuit) -> float:
    """Compute the *liveness* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the liveness feature for this circuit.
    """
    return supermarq.converters.compute_liveness_with_qiskit(
        (supermarq.converters.cirq_to_qiskit(circuit))
    )


def compute_parallelism(circuit: cirq.Circuit) -> float:
    """Compute the *parallelism* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the parallelism feature for this circuit.
    """
    return supermarq.converters.compute_parallelism_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )


def compute_measurement(circuit: cirq.Circuit) -> float:
    """Compute the *measurement* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the measurement feature for this circuit.
    """
    return supermarq.converters.compute_measurement_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )


def compute_entanglement(circuit: cirq.Circuit) -> float:
    """Compute the *entanglement* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the entanglement feature for this circuit.
    """
    return supermarq.converters.compute_entanglement_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )


def compute_depth(circuit: cirq.Circuit) -> float:
    """Compute the *depth* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the depth feature for this circuit.
    """
    return supermarq.converters.compute_depth_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )
