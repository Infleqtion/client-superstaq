# pylint: disable=missing-function-docstring,missing-class-docstring
from unittest.mock import MagicMock, patch

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def test_default_options() -> None:
    provider = qss.SuperstaQProvider(api_key="MY_TOKEN")
    backend = qss.SuperstaQBackend(provider=provider, target="ibmq_qasm_simulator")

    assert qiskit.providers.Options(shots=1000) == backend._default_options()


def test_validate_target() -> None:
    provider = qss.SuperstaQProvider(api_key="123")
    with pytest.raises(ValueError, match="does not have a valid string format"):
        qss.SuperstaQBackend(provider=provider, target="invalid_target")

    with pytest.raises(ValueError, match="does not have a valid target device type"):
        qss.SuperstaQBackend(provider=provider, target="ibmq_invalid_device")

    with pytest.raises(ValueError, match="does not have a valid target prefix"):
        qss.SuperstaQBackend(provider=provider, target="invalid_test_qpu")


def test_run() -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])

    backend = qss.SuperstaQProvider(api_key="123").get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaQClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=qc, shots=1000)
        expected = qss.SuperstaQJob(backend, "job_id")
        assert answer == expected

    with pytest.raises(ValueError, match="Circuit has no measurements to sample"):
        qc.remove_final_measurements()
        backend.run(qc, shots=1000)


def test_multi_circuit_run() -> None:
    qc1 = qiskit.QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)

    qc2 = qiskit.QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])

    backend = qss.SuperstaQProvider(api_key="123").get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaQClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=[qc1, qc2], shots=1000)
        expected = qss.SuperstaQJob(backend, "job_id")
        assert answer == expected


def test_multi_arg_run() -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])

    backend = qss.SuperstaQProvider(api_key="123").get_backend("ss_example_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaQClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        answer = backend.run(circuits=qc, shots=1000, method="fake_method", options={"test": "123"})
        expected = qss.SuperstaQJob(backend, "job_id")
        assert answer == expected


def test_eq() -> None:
    provider = qss.SuperstaQProvider(api_key="123")

    backend1 = provider.get_backend("ibmq_qasm_simulator")
    assert backend1 != 3

    backend2 = provider.get_backend("ibmq_athens_qpu")
    assert backend1 != backend2

    backend3 = provider.get_backend("ibmq_qasm_simulator")
    assert backend1 == backend3


@patch("requests.post")
def test_compile(mock_post: MagicMock) -> None:
    # AQT compile
    provider = qss.SuperstaQProvider(api_key="MY_TOKEN")
    backend = provider.get_backend("aqt_keysight_qpu")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[1, 4]]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }
    out = backend.compile(qc)
    assert out.circuit == qc
    assert out.final_logical_to_physical == {1: 4}
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = backend.compile([qc], atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{1: 4}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]], [[]]]),
    }
    out = backend.compile([qc, qc], test_options="yes")
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    # AQT ECA compile
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }

    out = backend.compile(
        qc, num_equivalent_circuits=1, random_seed=1234, atol=1e-2, test_options="yes"
    )
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")
