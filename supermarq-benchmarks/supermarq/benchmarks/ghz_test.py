# pylint: disable=missing-function-docstring,missing-class-docstring
import cirq

from supermarq.benchmarks.ghz import GHZ


def test_ghz_circuit() -> None:
    ghz = GHZ(3)
    cirq_circuit = ghz.cirq_circuit()
    if isinstance(cirq_circuit, cirq.Circuit):
        assert len(cirq_circuit.all_qubits()) == 3
    assert ghz.qiskit_circuit().num_qubits == 3


def test_ghz_score() -> None:
    ghz = GHZ(3)
    assert ghz.score({"000": 500, "111": 500}) == 1
