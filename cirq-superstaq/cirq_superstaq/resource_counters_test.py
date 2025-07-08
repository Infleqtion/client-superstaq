from __future__ import annotations

import cirq
import cirq_superstaq as css
import numpy as np

import superstaq as ss


def test_resource_counters() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.Z(qubits[2]),
        css.ParallelRGate(np.pi / 2, np.pi / 2, len(qubits)).on(*qubits),
        cirq.ISWAP(qubits[0], qubits[1]),
        cirq.CZ(qubits[1], qubits[2]),
        cirq.TOFFOLI(qubits[0], qubits[1], qubits[2]),
    )

    assert ss.resource_counters.num_single_qubit_gates(circuit) == 3
    assert ss.resource_counters.num_two_qubit_gates(circuit) == 2
    assert ss.resource_counters.num_phased_xpow_subgates(circuit) == 2
    assert ss.resource_counters.num_global_ops(circuit) == 2
    assert np.isclose(ss.resource_counters.total_global_rgate_pi_time(circuit), 0.5)
