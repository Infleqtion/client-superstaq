# pylint: disable=missing-function-docstring,missing-class-docstring
import cirq

import supermarq
from supermarq.benchmarks.qaoa_fermionic_swap_proxy import QAOAFermionicSwapProxy


def test_qaoa_circuit() -> None:
    """Test the circuit generation function."""
    qaoa = QAOAFermionicSwapProxy(4)
    assert len(qaoa.circuit().all_qubits()) == 4
    assert (
        len(
            list(qaoa.circuit().findall_operations(lambda op: isinstance(op.gate, type(cirq.CNOT))))
        )
        == 18
    )


def test_qaoa_score() -> None:
    """Test the score evaluation function."""
    qaoa = QAOAFermionicSwapProxy(4)
    # Reverse bitstring ordering due to SWAP network
    raw_counts = supermarq.simulation.get_ideal_counts(qaoa.circuit())
    ideal_counts = {bitstring[::-1]: probability for bitstring, probability in raw_counts.items()}
    assert qaoa.score({k[::-1]: v for k, v in ideal_counts.items()}) > 0.99
