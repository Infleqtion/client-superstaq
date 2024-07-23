# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import cirq
import qiskit

import supermarq


def test_cirq_to_qiskit() -> None:
    cirq_circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.CX(cirq.LineQubit(0), cirq.LineQubit(1))
    )
    qiskit_circuit = qiskit.QuantumCircuit(2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    assert supermarq.converters.cirq_to_qiskit(cirq_circuit) == qiskit_circuit


def test_compute_parallelism_with_qiskit() -> None:
    qiskit_circuit = qiskit.QuantumCircuit(2)
    assert supermarq.converters.compute_parallelism_with_qiskit(qiskit_circuit) == 0
