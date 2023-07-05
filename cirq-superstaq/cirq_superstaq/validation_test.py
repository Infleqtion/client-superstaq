# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest
import cirq

import cirq_superstaq as css

def test_validate_cirq_circuits() -> None:
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.service._validate_cirq_circuits("circuit_invalid")

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
        "sequence of `cirq.Circuit` instances.",
    ):
        css.service._validate_cirq_circuits([circuit, "circuit_invalid"])