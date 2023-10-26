# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""

import cirq
import numpy as np
import pytest

import cirq_superstaq as css


@pytest.fixture
def service() -> css.Service:
    return css.Service()


@pytest.mark.skipif(
    "cq_hilbert_qpu" not in css.Service().get_targets()["compile-and-run"],
    reason="Can't be executed when Hilbert is set to not accept jobs",
)
def test_submit_to_hilbert_qubit_sorting(service: css.Service) -> None:
    """Regression test for https://github.com/Infleqtion/client-superstaq/issues/776"""
    target = "cq_hilbert_qpu"
    num_qubits = service.target_info(target)["num_qubits"]

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0.0, 24).on(*qubits),
        cirq.rz(np.pi).on(qubits[2]),
        css.ParallelRGate(-np.pi / 2, 0.0, 24).on(*qubits),
        cirq.measure(*qubits),
    )

    job = service.create_job(circuit, repetitions=100, verbatim=True, route=False, target=target)
    counts = job.counts(0)
    assert sum(counts.values()) == 100
    assert max(counts, key=counts.__getitem__) == "001" + ("0" * (num_qubits - 3))
