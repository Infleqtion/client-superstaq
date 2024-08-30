# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import re
from unittest.mock import MagicMock, create_autospec

import cirq
import pytest

import cirq_superstaq as css


def test_validate_cirq_circuits() -> None:
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.validation.validate_cirq_circuits("circuit_invalid")

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.validation.validate_cirq_circuits([circuit, "circuit_invalid"])

    with pytest.raises(ValueError, match="Circuit has no measurements to sample"):
        css.validation.validate_cirq_circuits(circuit, require_measurements=True)


def test_validate_qubit_type() -> None:
    invalid_qubit_type = MagicMock(spec=str)  # E.g., in practice, `cirq_rigetti.AspenQubit`
    mock_cirq_circuit = create_autospec(cirq.Circuit, spec_set=True)
    mock_cirq_circuit.all_qubits.return_value = frozenset({invalid_qubit_type, cirq.LineQubit(0)})
    q0 = cirq.LineQubit(0)
    q1, q2 = cirq.NamedQubit.range(2, prefix="q")
    q3 = cirq.GridQubit(0, 0)
    q4 = cirq.q(4)
    valid_qubits = frozenset({q0, q1, q2, q3, q4})
    valid_circuit = cirq.Circuit(cirq.H(q) for q in valid_qubits)
    valid_circuit += cirq.measure(*valid_qubits)

    css.validation.validate_qubit_types(valid_circuit)
    with pytest.raises(
        TypeError,
        match=re.escape("Input circuit(s) contains unsupported qubit types:"),
    ):
        css.validation.validate_cirq_circuits([mock_cirq_circuit, valid_circuit])
