from __future__ import annotations

import cirq
import networkx as nx
import numpy as np


def compute_communication(circuit: cirq.Circuit) -> float:
    """Compute the *communication* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the communication feature for this circuit.
    """
    qubits = sorted(circuit.all_qubits(), key=lambda q: str(q))
    num_qubits = len(qubits)
    qubit_to_index = {q: i for i, q in enumerate(qubits)}

    # Build interaction graph: nodes are qubits, edges are two-qubit gates
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))

    for moment in circuit:
        for op in moment.operations:
            qargs = list(op.qubits)
            if len(qargs) == 2:
                idx1 = qubit_to_index[qargs[0]]
                idx2 = qubit_to_index[qargs[1]]
                graph.add_edge(idx1, idx2)

    # Sum degrees of all qubits
    degree_sum = sum(graph.degree(n) for n in graph.nodes)

    # Degree of a complete graph: num_qubits * (num_qubits - 1)
    return degree_sum / (num_qubits * (num_qubits - 1))


def compute_liveness(circuit: cirq.Circuit) -> float:
    """Compute the *liveness* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the liveness feature for this circuit.
    """
    moments = circuit.moments
    qubits = sorted(
        circuit.all_qubits(),
        key=lambda q: (q.row, q.col) if hasattr(q, "row") and hasattr(q, "col") else str(q),
    )
    num_qubits = len(qubits)
    depth = len(moments)
    # Map qubits to row indices
    qubit_indices = {q: i for i, q in enumerate(qubits)}
    activity_matrix = np.zeros((num_qubits, depth), dtype=int)

    for i, moment in enumerate(moments):
        for op in moment.operations:
            for q in op.qubits:
                activity_matrix[qubit_indices[q], i] = 1
    return np.sum(activity_matrix) / (num_qubits * depth)


def compute_parallelism(circuit: cirq.Circuit) -> float:
    """Compute the *parallelism* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the parallelism feature for this circuit.
    """
    num_qubits = len(circuit.all_qubits())
    if num_qubits <= 1:
        return 0
    depth = len(circuit.moments)
    if depth == 0:
        return 0
    num_gates = sum(1 for moment in circuit for _ in moment.operations)
    return max(((num_gates / depth) - 1) / (num_qubits - 1), 0)


def compute_measurement(circuit: cirq.Circuit) -> float:
    """Compute the *measurement* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the measurement feature for this circuit.
    """
    reset_moments = 0
    depth = len(circuit.moments)

    for moment in circuit.moments:
        reset_present = False
        for op in moment.operations:
            # Check for mid-circuit measurement: cirq.MeasurementGate not at the end
            if isinstance(op.gate, cirq.ResetChannel) or getattr(op.gate, "name", None) == "reset":
                reset_present = True
        if reset_present:
            reset_moments += 1
    return reset_moments / depth


def compute_entanglement(circuit: cirq.Circuit) -> float:
    """Compute the *entanglement* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the entanglement feature for this circuit.
    """
    two_qubit_gates = 0
    total_gates = 0

    for moment in circuit.moments:
        for op in moment.operations:
            # Count only gate operations (ignore measurements, etc)
            if isinstance(op.gate, cirq.Gate):
                total_gates += 1
                if len(op.qubits) == 2:
                    two_qubit_gates += 1
    return two_qubit_gates / total_gates


def compute_depth(circuit: cirq.Circuit) -> float:
    """Compute the *depth* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the depth feature for this circuit.
    """
    two_qubit_gates_per_moment = []
    for _, moment in enumerate(circuit.moments):
        two_qubit_gates = []
        for op in moment.operations:
            if isinstance(op.gate, cirq.Gate) and len(op.qubits) == 2:
                two_qubit_gates.append(op)
        two_qubit_gates_per_moment.append(two_qubit_gates)
    total_two_qubit_gates = sum(len(x) for x in two_qubit_gates_per_moment)

    # Find critical path: for each qubit, count the 2-qubit gates
    # it participates in along the longest path
    # Track the path length per qubit
    qubits = sorted(circuit.all_qubits(), key=lambda q: str(q))
    qubit_last_moment = {q: -1 for q in qubits}
    qubit_depths = {q: 0 for q in qubits}

    for i, moment in enumerate(circuit.moments):
        for op in moment.operations:
            if isinstance(op.gate, cirq.Gate) and len(op.qubits) == 2:
                for q in op.qubits:
                    if qubit_last_moment[q] + 1 <= i:
                        qubit_depths[q] += 1
                        qubit_last_moment[q] = i

    # The critical path is the maximum number of 2-qubit gates seen by any qubit
    critical_two_qubit_gates = max(qubit_depths.values()) if qubit_depths else 0
    if total_two_qubit_gates == 0:
        return 0.0
    return critical_two_qubit_gates / total_two_qubit_gates
