# pylint: disable=missing-function-docstring,missing-class-docstring
import cirq

import supermarq
from supermarq.benchmarks.qaoa_vanilla_proxy import QAOAVanillaProxy


def test_qaoa_circuit() -> None:
    qaoa = QAOAVanillaProxy(4)
    assert len(qaoa.circuit().all_qubits()) == 4
    assert (
        len(
            list(qaoa.circuit().findall_operations(lambda op: isinstance(op.gate, type(cirq.CNOT))))
        )
        == 12
    )


def test_qaoa_score() -> None:
    qaoa = QAOAVanillaProxy(4)
    assert qaoa.score(supermarq.simulation.get_ideal_counts(qaoa.circuit())) > 0.99
