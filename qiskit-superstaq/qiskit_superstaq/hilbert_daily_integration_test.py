# pylint: disable=missing-function-docstring,missing-class-docstring
"""Integration checks that run daily (via Github action) between client and prod server."""
import numpy as np
import pytest
import qiskit

import qiskit_superstaq as qss


@pytest.fixture
def provider() -> qss.SuperstaqProvider:
    return qss.SuperstaqProvider()


@pytest.mark.skipif(
    "cq_hilbert_qpu"
    not in qss.SuperstaqProvider()._client.get_targets()["superstaq_targets"]["compile-and-run"],
    reason="Can't be executed when Hilbert is set to not accept jobs",
)
def test_submit_to_hilbert_qubit_sorting(provider: qss.SuperstaqProvider) -> None:
    """Regression test for https://github.com/Infleqtion/client-superstaq/issues/776"""
    backend = provider.get_backend("cq_hilbert_qpu")

    num_qubits = backend.configuration().n_qubits

    gr = qiskit.circuit.library.GR(num_qubits, np.pi / 2, 0)
    grdg = qiskit.circuit.library.GR(num_qubits, -np.pi / 2, 0)

    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(gr, range(num_qubits))
    qc.rz(np.pi, 2)
    qc.append(grdg, range(num_qubits))
    qc.measure_all()

    job = backend.run(qc, 100, verbatim=True, route=False)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100
    assert max(counts, key=counts.__getitem__) == ("0" * (num_qubits - 3)) + "100"
