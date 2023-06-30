# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest
import qiskit

import qiskit_superstaq as qss


def test_validate_qiskit_circuits() -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a sequence "
        "of `qiskit.QuantumCircuit` instances.",
    ):
        qss.validation.validate_qiskit_circuits("invalid_qc_input")

    with pytest.raises(
        ValueError,
        match="Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a "
        "sequence of `qiskit.QuantumCircuit` instances.",
    ):
        qss.validation.validate_qiskit_circuits([qc, "invalid_qc_input"])
