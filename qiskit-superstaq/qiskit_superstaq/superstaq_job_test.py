# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import datetime
import json
import uuid
from typing import TYPE_CHECKING
from unittest import mock

import general_superstaq as gss
import pytest
import qiskit
import requests

import qiskit_superstaq as qss

if TYPE_CHECKING:
    from qiskit_superstaq.conftest import MockSuperstaqProvider


def mock_response(status_str: str) -> dict[str, str | int | dict[str, int] | None]:
    """Mock response for requests.

    Args:
        status_str: the value for the status key.

    Returns:
        A mock response.
    """
    return {"status": status_str, "samples": {"11": 50, "10": 50}, "shots": 100}


def _mocked_request_response(content: object) -> requests.Response:
    response = requests.Response()
    response.status_code = requests.codes.OK
    response._content = json.dumps(content).encode()
    return response


def patched_requests(*contents: object) -> mock._patch[mock.Mock]:
    """Mocks all server responses with the given sequence of content objects.

    Args:
        contents: The JSON contents to return for each request.

    Returns:
        A mock patch that returns the provided content.
    """
    responses = [_mocked_request_response(val) for val in contents]
    return mock.patch("requests.Session.request", side_effect=responses)


@pytest.fixture
def backend(fake_superstaq_provider: MockSuperstaqProvider) -> qss.SuperstaqBackend:
    """Fixture for qiskit_superstaq backend.

    Args:
        fake_superstaq_provider: the mocked SuperstaqProvider.

    Returns:
        A mocked SuperstaqBackend.
    """
    return fake_superstaq_provider.get_backend("ss_example_qpu")


@pytest.fixture
def backendV3(fake_superstaq_providerV3: MockSuperstaqProvider) -> qss.SuperstaqBackend:
    """Fixture for qiskit_superstaq backend.

    Args:
        fake_superstaq_providerV3: the mocked SuperstaqProvider.

    Returns:
        A mocked SuperstaqBackend.
    """
    return fake_superstaq_providerV3.get_backend("ss_example_qpu")


def job_dictV3(n_circuits: int = 1) -> dict[str, object]:
    """Fixture for a standard, completed single v0.3.0 job result.

    Args:
        n_circuits: the number of circuits to include in the dictionary.

    Returns:
        A dictionary containing commonly expected job data.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return {
        "job_type": "simulate",
        "statuses": ["completed"] * n_circuits,
        "status_messages": [None] * n_circuits,
        "user_email": "test@email.com",
        "target": "ss_unconstrained_simulator",
        "provider_id": ["provider_id"] * n_circuits,
        "num_circuits": n_circuits,
        "compiled_circuits": [None] * n_circuits,
        "input_circuits": [qss.serialize_circuits(circuit)] * n_circuits,
        "circuit_type": "cirq",
        "counts": [{"11": 1}] * n_circuits,
        "results_dicts": [] * n_circuits,
        "shots": [1] * n_circuits,
        "dry_run": True,
        "submission_timestamp": str(datetime.datetime.now()),
        "last_updated_timestamp": [str(datetime.datetime.now())] * n_circuits,
        "initial_logical_to_physicals": [{0: 0, 1: 1}] * n_circuits,
        "final_logical_to_physicals": [{0: 0, 1: 1}] * n_circuits,
        "logical_qubits": ["0", "1"],
        "physical_qubits": ["0", "1"],
        "tags": ["some", "tags"],
        "metadata": {"foo": "bar"},
    }


def modifiy_job_result(base_result: dict[str, object], **kwargs: object) -> dict[str, object]:
    """Extends and updates `base_result` with passed `kwargs`.

    Args:
        base_result: Base job result to modify.
        kwargs: Additional keyword args to add or update existing values in `base_result`.

    Returns:
        An updated dictionary based on the `kwargs` passed.
    """
    return {**base_result, **kwargs}


def test_wait_for_results(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    jobs = qss.SuperstaqJob(backend=backend, job_id="123abc,456def")

    with patched_requests({"123abc": mock_response("Done")}):
        assert job._wait_for_results(timeout=backend._provider._client.max_retry_seconds) == [
            mock_response("Done")
        ]
    with patched_requests({"123abc": mock_response("Done"), "456def": mock_response("Done")}):
        assert jobs._wait_for_results(timeout=backend._provider._client.max_retry_seconds) == [
            mock_response("Done"),
            mock_response("Done"),
        ]


def test_wait_for_resultsV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    response_dict = {str(job._job_id): job_dictV3()}
    with patched_requests(response_dict):
        assert job._wait_for_results(
            timeout=backendV3._provider._client.max_retry_seconds
        ).statuses == ["completed"]

    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    response_dict = {str(job._job_id): job_dictV3(2)}
    with patched_requests(response_dict):
        assert job._wait_for_results(
            timeout=backendV3._provider._client.max_retry_seconds
        ).statuses == ["completed", "completed"]


def test_cancel(backend: qss.SuperstaqBackend) -> None:
    multi_job = qss.SuperstaqJob(backend=backend, job_id="123abc,456def,789abc")
    with mock.patch("requests.Session.post", return_value=mock.MagicMock(ok=True)) as mock_post:
        qss.SuperstaqJob(backend=backend, job_id="123abc").cancel()
        multi_job.cancel(0)
        multi_job.cancel()
        assert mock_post.call_count == 3


def test_cancelV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    with mock.patch("requests.Session.put") as mock_put:
        mock_put.return_value.json.return_value = {"succeeded": ["circuit"], "message": "message"}
        job.cancel()


def test_timeout(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")

    with patched_requests(
        {"123abc": mock_response("Queued")},
        {"123abc": mock_response("Queued")},
        {"123abc": mock_response("Done")},
    ) as mocked_get_job:
        assert job._wait_for_results(
            timeout=backend._provider._client.max_retry_seconds, wait=0.0
        ) == [mock_response("Done")]
        assert mocked_get_job.call_count == 3


def test_timeoutV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    completed_dict = job_dictV3()
    queue_dict = modifiy_job_result(completed_dict, statuses=["awaiting_submission"])

    with patched_requests(
        {str(job._job_id): queue_dict},
        {str(job._job_id): queue_dict},
        {str(job._job_id): completed_dict},
    ) as mocked_get_job:
        assert job._wait_for_results(
            timeout=backendV3._provider._client.max_retry_seconds, wait=0.0
        ).statuses == ["completed"]
        assert mocked_get_job.call_count == 3

    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    with (
        mock.patch("time.sleep", return_value=None),
        patched_requests(
            {str(job._job_id): queue_dict},
            {str(job._job_id): queue_dict},
            {str(job._job_id): queue_dict},
            {str(job._job_id): completed_dict},
        ) as mocked_get_job,
        pytest.raises(TimeoutError, match=r"Timed out while waiting for results."),
    ):
        job._wait_for_results(timeout=10, wait=8)


def test_result(backend: qss.SuperstaqBackend) -> None:
    qc = qiskit.QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")

    response = mock_response("Done")
    response["input_circuit"] = qss.serialize_circuits(qc)
    response["compiled_circuit"] = qss.serialize_circuits(qc)
    response["samples"] = {"110": 30, "100": 50, "111": 20}
    expected_results = [
        {"success": True, "shots": 100, "data": {"counts": {"110": 30, "100": 50, "111": 20}}}
    ]

    expected = qiskit.result.Result.from_dict(
        {
            "results": expected_results,
            "qobj_id": -1,
            "backend_name": "ss_example_qpu",
            "backend_version": gss.API_VERSION,
            "success": True,
            "job_id": "123abc",
        }
    )

    with patched_requests({"123abc": response}):
        ans = (
            job.result(index=0),
            job.result(index=0, qubit_indices=[0]),
            job.result(index=0, qubit_indices=[2]),
            job.result(index=0, qubit_indices=[1, 2]),
        )
        assert ans[0].backend_name == expected.backend_name
        assert ans[0].job_id == expected.job_id
        assert ans[0].get_counts() == {"011": 30, "001": 50, "111": 20}
        assert ans[1].get_counts() == {"1": 100}
        assert ans[2].get_counts() == {"0": 80, "1": 20}
        assert ans[3].get_counts() == {"01": 30, "00": 50, "11": 20}

    multi_job = qss.SuperstaqJob(backend=backend, job_id="123abc,456xyz")
    with patched_requests({"123abc": response, "456xyz": response}):
        assert multi_job.result().get_counts() == [
            {"011": 30, "001": 50, "111": 20},
            {"011": 30, "001": 50, "111": 20},
        ]
        assert multi_job.result(index=0).get_counts() == {"011": 30, "001": 50, "111": 20}


def test_resultV3(backendV3: qss.SuperstaqBackend) -> None:
    qc = qiskit.QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])

    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))

    job_dict = modifiy_job_result(
        job_dictV3(),
        input_circuits=[qss.serialize_circuits(qc)],
        compiled_circuits=[qss.serialize_circuits(qc)],
        counts=[{"110": 30, "100": 50, "111": 20}],
        shots=[100],
    )
    results_list = [
        {"success": True, "shots": 100, "data": {"counts": {"011": 30, "001": 50, "111": 20}}}
    ]
    expected_result = qiskit.result.Result.from_dict(
        {
            "results": results_list,
            "qobj_id": -1,
            "backend_name": "ss_example_qpu",
            "backend_version": "n/a",
            "success": True,
            "job_id": uuid.UUID(int=42),
        }
    )

    with patched_requests({str(job._job_id): job_dict}):
        ans = (
            job.result(index=0),
            job.result(index=0, qubit_indices=[0]),
            job.result(index=0, qubit_indices=[2]),
            job.result(index=0, qubit_indices=[1, 2]),
        )
        assert ans[0].backend_name == expected_result.backend_name
        assert ans[0].job_id == expected_result.job_id
        assert ans[0].get_counts() == expected_result.get_counts()
        assert ans[1].get_counts() == {"1": 100}
        assert ans[2].get_counts() == {"0": 80, "1": 20}
        assert ans[3].get_counts() == {"01": 30, "00": 50, "11": 20}

    multi_job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))

    job_dict = modifiy_job_result(
        job_dictV3(2),
        input_circuits=[qss.serialize_circuits(qc)] * 2,
        compiled_circuits=[qss.serialize_circuits(qc)] * 2,
        counts=[{"110": 30, "100": 50, "111": 20}] * 2,
        shots=[100] * 2,
    )

    with patched_requests({str(multi_job._job_id): job_dict}):
        assert multi_job.result().get_counts() == [
            {"011": 30, "001": 50, "111": 20},
            {"011": 30, "001": 50, "111": 20},
        ]
        assert multi_job.result(index=0).get_counts() == {"011": 30, "001": 50, "111": 20}


def test_counts_arranged(backend: qss.SuperstaqBackend) -> None:
    # Test case: len(qiskit.ClassicalRegister()) = len(qiskit.QuantumRegister())
    qc1 = qiskit.QuantumCircuit(qiskit.QuantumRegister(4), qiskit.ClassicalRegister(4))
    qc1.measure([2, 3], [2, 3])
    qc2 = qiskit.QuantumCircuit(qiskit.QuantumRegister(5), qiskit.ClassicalRegister(5))
    qc2.measure([0, 1, 2, 4], [0, 1, 2, 4])

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc")
    job._job_info["123abc"] = {
        "status": "Done",
        "samples": {"00": 70, "11": 30},
        "shots": 100,
        "input_circuit": qss.serialization.serialize_circuits(qc1),
        "compiled_circuit": qss.serialization.serialize_circuits(qc1),
    }
    job._job_info["456abc"] = {
        "status": "Done",
        "samples": {"1101": 40, "0001": 60},
        "shots": 100,
        "input_circuit": qss.serialization.serialize_circuits(qc2),
        "compiled_circuit": qss.serialization.serialize_circuits(qc2),
    }
    counts = job.result().get_counts()
    assert counts == [{"0000": 70, "1100": 30}, {"10011": 40, "10000": 60}]

    # Test case: len(qiskit.ClassicalRegister()) < len(qiskit.QuantumRegister())
    qc3 = qiskit.QuantumCircuit(qiskit.QuantumRegister(8), qiskit.ClassicalRegister(5))
    qc3.x(0)
    qc3.h(1)
    qc3.measure(1, 1)
    qc3.h(1)
    qc3.cx(1, 2)
    qc3.measure([6, 2, 0], [3, 0, 4])
    job = qss.SuperstaqJob(backend=backend, job_id="987def")
    job._job_info["987def"] = {
        "status": "Done",
        "samples": {"1101": 26, "0001": 36, "1001": 19, "0101": 19},
        "shots": 100,
        "input_circuit": qss.serialization.serialize_circuits(qc3),
        "compiled_circuit": qss.serialization.serialize_circuits(qc3),
    }
    counts = job.result(0).get_counts()
    assert counts == {"10011": 26, "10010": 19, "10001": 19, "10000": 36}

    # Test case: len(qiskit.ClassicalRegister()) > len(qiskit.QuantumRegister())
    qc4 = qiskit.QuantumCircuit(qiskit.QuantumRegister(3), qiskit.ClassicalRegister(5))
    qc4.h(1)
    qc4.x(2)
    qc4.measure([0, 1, 2], [2, 4, 1])
    job = qss.SuperstaqJob(backend=backend, job_id="789cba")
    job._job_info["789cba"] = {
        "status": "Done",
        "samples": {"100": 50, "101": 50},
        "shots": 100,
        "input_circuit": qss.serialization.serialize_circuits(qc4),
        "compiled_circuit": qss.serialization.serialize_circuits(qc4),
    }
    counts = job.result(0).get_counts()
    assert counts == {"00010": 50, "10010": 50}


def test_counts_arrangedV3(backendV3: qss.SuperstaqBackend) -> None:
    # Test case: len(qiskit.ClassicalRegister()) = len(qiskit.QuantumRegister())
    qc1 = qiskit.QuantumCircuit(qiskit.QuantumRegister(4), qiskit.ClassicalRegister(4))
    qc1.measure([2, 3], [2, 3])
    qc2 = qiskit.QuantumCircuit(qiskit.QuantumRegister(5), qiskit.ClassicalRegister(5))
    qc2.measure([0, 1, 2, 4], [0, 1, 2, 4])

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job_dict = job_dictV3(2)
    job_dict["statuses"] = ["completed"] * 2
    job_dict["counts"] = [{"00": 70, "11": 30}, {"1101": 40, "0001": 60}]
    job_dict["input_circuits"] = [
        qss.serialization.serialize_circuits(qc1),
        qss.serialization.serialize_circuits(qc2),
    ]
    job_dict["compiled_circuits"] = [
        qss.serialization.serialize_circuits(qc1),
        qss.serialization.serialize_circuits(qc2),
    ]
    job_dict["shots"] = [100] * 2

    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()

    counts = job.result().get_counts()
    assert counts == [{"0000": 70, "1100": 30}, {"10011": 40, "10000": 60}]

    # Test case: len(qiskit.ClassicalRegister()) < len(qiskit.QuantumRegister())
    qc3 = qiskit.QuantumCircuit(qiskit.QuantumRegister(8), qiskit.ClassicalRegister(5))
    qc3.x(0)
    qc3.h(1)
    qc3.measure(1, 1)
    qc3.h(1)
    qc3.cx(1, 2)
    qc3.measure([6, 2, 0], [3, 0, 4])

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["completed"]
    job_dict["counts"] = [{"1101": 26, "0001": 36, "1001": 19, "0101": 19}]
    job_dict["input_circuits"] = [qss.serialization.serialize_circuits(qc3)]
    job_dict["compiled_circuits"] = [qss.serialization.serialize_circuits(qc3)]
    job_dict["shots"] = [100]

    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    counts = job.result().get_counts()
    assert counts == {"10011": 26, "10010": 19, "10001": 19, "10000": 36}

    # Test case: len(qiskit.ClassicalRegister()) > len(qiskit.QuantumRegister())
    qc4 = qiskit.QuantumCircuit(qiskit.QuantumRegister(3), qiskit.ClassicalRegister(5))
    qc4.h(1)
    qc4.x(2)
    qc4.measure([0, 1, 2], [2, 4, 1])

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["completed"]
    job_dict["counts"] = [{"100": 50, "101": 50}]
    job_dict["input_circuits"] = [qss.serialization.serialize_circuits(qc4)]
    job_dict["compiled_circuits"] = [qss.serialization.serialize_circuits(qc4)]
    job_dict["shots"] = [100]

    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    counts = job.result(0).get_counts()
    assert counts == {"00010": 50, "10010": 50}


def test_get_clbit_indices(backend: qss.SuperstaqBackend) -> None:
    response = mock_response("Done")
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    response["input_circuit"] = qss.serialization.serialize_circuits(qc)
    response["compiled_circuit"] = qss.serialization.serialize_circuits(qc)

    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with patched_requests({"123abc": response}):
        returned_meas_list = job._get_clbit_indices(index=0)
        assert returned_meas_list == [0, 1]


def test_get_clbit_indicesV3(backendV3: qss.SuperstaqBackend) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["completed"]
    job_dict["input_circuit"] = [qss.serialization.serialize_circuits(qc)]

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job._job_info = gss.models.JobData(**job_dict)

    returned_meas_list = job._get_clbit_indices(index=0)
    assert returned_meas_list == [0, 1]


def test_get_num_clbits(backend: qss.SuperstaqBackend) -> None:
    response = mock_response("Done")
    qc = qiskit.QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 0)
    response["input_circuit"] = qss.serialize_circuits(qc)
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with patched_requests({"123abc": response}):
        assert job._get_num_clbits(index=0) == 2


def test_get_num_clbitsV3(backendV3: qss.SuperstaqBackend) -> None:
    qc = qiskit.QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 0)

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["completed"]
    job_dict["input_circuit"] = [qss.serialization.serialize_circuits(qc)]

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job._job_info = gss.models.JobData(**job_dict)

    assert job._get_num_clbits(index=0) == 2


def test_arrange_counts(backend: qss.SuperstaqBackend) -> None:
    sample_counts = {"011": 100, "001": 25, "111": 100, "101": 25}
    sample_meas_bit_indices = [0, 2, 4]
    sample_num_clbits = 5
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    assert job._arrange_counts(sample_counts, sample_meas_bit_indices, sample_num_clbits) == {
        "00101": 100,
        "00001": 25,
        "10101": 100,
        "10001": 25,
    }


def test_arrange_countsV3(backendV3: qss.SuperstaqBackend) -> None:
    sample_counts = {"011": 100, "001": 25, "111": 100, "101": 25}
    sample_meas_bit_indices = [0, 2, 4]
    sample_num_clbits = 5
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    assert job._arrange_counts(sample_counts, sample_meas_bit_indices, sample_num_clbits) == {
        "00101": 100,
        "00001": 25,
        "10101": 100,
        "10001": 25,
    }


def test_check_if_stopped(backend: qss.SuperstaqBackend) -> None:
    for status in ("Cancelled", "Failed"):
        job = qss.SuperstaqJob(backend=backend, job_id="123abc")
        job._overall_status = status
        with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match=status):
            job._check_if_stopped()


def test_check_if_stoppedV3(backendV3: qss.SuperstaqBackend) -> None:
    for status in (gss.models.CircuitStatus.CANCELLED, gss.models.CircuitStatus.FAILED):
        job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
        job._overall_status = status
        with pytest.raises(
            gss.SuperstaqUnsuccessfulJobException,
            match=f"Job {uuid.UUID(int=42)} terminated with status {status.value}",
        ):
            job._check_if_stopped()


def test_refresh_job(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response = {
        "123abc": mock_response("Running"),
        "456abc": mock_response("Queued"),
        "789abc": mock_response("Done"),
    }

    assert job._overall_status == "Submitted"

    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "Queued"

    response["456abc"] = mock_response("Done")
    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "Running"

    response["123abc"] = mock_response("Done")
    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "Done"

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response["123abc"] = mock_response("Failed")
    with patched_requests(response):
        assert job._overall_status == "Submitted"
        job._refresh_job()
        assert job._overall_status == "Failed"

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response["123abc"] = mock_response("Cancelled")
    with patched_requests(response):
        assert job._overall_status == "Submitted"
        job._refresh_job()
        assert job._overall_status == "Cancelled"


def test_refresh_jobV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(3)
    job_dict["statuses"] = ["running", "awaiting_submission", "completed"]
    response = {str(uuid.UUID(int=42)): job_dict}

    assert job._overall_status == "received"

    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["running", "completed", "completed"]
    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "running"

    job_dict["statuses"] = ["completed", "completed", "completed"]
    with patched_requests(response):
        job._refresh_job()
        assert job._overall_status == "completed"

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["failed"]

    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        assert job._overall_status == "received"
        job._refresh_job()
        assert job._overall_status == "failed"

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["cancelled"]

    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        assert job._overall_status == "received"
        job._refresh_job()
        assert job._overall_status == "cancelled"


def test_status(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response = {
        "123abc": mock_response("Running"),
        "456abc": mock_response("Submitted"),
        "789abc": mock_response("Done"),
    }
    with patched_requests(response):
        assert job.status() == qiskit.providers.JobStatus.INITIALIZING
        assert job.status(index=2) == qiskit.providers.JobStatus.DONE

    response["456abc"] = mock_response("Queued")
    with patched_requests(response):
        assert job.status(index=2) == qiskit.providers.JobStatus.DONE
        assert job.status() == qiskit.providers.JobStatus.QUEUED

    response["456abc"] = mock_response("Done")
    with patched_requests(response):
        assert job.status(index=2) == qiskit.providers.JobStatus.DONE
        assert job.status() == qiskit.providers.JobStatus.RUNNING

    response["123abc"] = mock_response("Done")
    with patched_requests(response):
        assert job.status() == qiskit.providers.JobStatus.DONE

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response["123abc"] = mock_response("Failed")
    with patched_requests(response):
        assert job.status() == qiskit.providers.JobStatus.ERROR

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    response["123abc"] = mock_response("Cancelled")
    with patched_requests(response):
        assert job.status() == qiskit.providers.JobStatus.CANCELLED
        assert job.status(0) == qiskit.providers.JobStatus.CANCELLED


def test_statusV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(3)
    job_dict["statuses"] = ["running", "received", "completed"]
    response = {str(uuid.UUID(int=42)): job_dict}

    with patched_requests(response, response):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.INITIALIZING
        assert job.status(index=2) == qiskit.providers.jobstatus.JobStatus.DONE

    job_dict["statuses"] = ["running", "awaiting_submission", "completed"]
    with patched_requests(response, response):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.QUEUED
        assert job.status(index=2) == qiskit.providers.jobstatus.JobStatus.DONE

    job_dict["statuses"] = ["running", "completed", "completed"]
    with patched_requests(response, response):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.RUNNING
        assert job.status(index=2) == qiskit.providers.jobstatus.JobStatus.DONE

    job_dict["statuses"] = ["completed", "completed", "completed"]
    with patched_requests(response, response):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.DONE
        assert job.status(index=2) == qiskit.providers.jobstatus.JobStatus.DONE

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["failed"]

    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.ERROR

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(1)
    job_dict["statuses"] = ["cancelled"]

    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        assert job.status() == qiskit.providers.jobstatus.JobStatus.CANCELLED


def test_update_status_queue_info(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456abc,789abc")

    for job_id in job._job_id.split(","):
        job._job_info[job_id] = mock_response("Done")

    job._refresh_job()
    assert job._overall_status == "Done"

    mock_statuses = [
        mock_response("Queued"),
        mock_response("Cancelled"),
        mock_response("Cancelled"),
    ]
    for index, job_id in enumerate(job._job_id.split(",")):
        job._job_info[job_id] = mock_statuses[index]
    job._update_status_queue_info()
    assert job._overall_status == "Queued"

    mock_statuses = [
        mock_response("Cancelled"),
        mock_response("Cancelled"),
        mock_response("Queued"),
    ]
    for index, job_id in enumerate(job._job_id.split(",")):
        job._job_info[job_id] = mock_statuses[index]
    job._update_status_queue_info()
    assert job._overall_status == "Queued"

    mock_statuses = [mock_response("Done"), mock_response("Done"), mock_response("Failed")]
    for index, job_id in enumerate(job._job_id.split(",")):
        job._job_info[job_id] = mock_statuses[index]
    job._update_status_queue_info()
    assert job._overall_status == "Failed"


def test_update_status_queue_infoV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))

    job_dict = job_dictV3(3)
    job_dict["statuses"] = ["completed"] * 3
    job._job_info = gss.models.JobData(**job_dict)

    job._update_status_queue_info()
    assert job._overall_status == "completed"

    job_dict["statuses"] = ["awaiting_submission", "cancelled", "cancelled"]
    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["cancelled", "cancelled", "awaiting_submission"]
    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["completed", "completed", "failed"]
    job._job_info = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "failed"


def test_get_circuit(backend: qss.SuperstaqBackend) -> None:
    test_job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with pytest.raises(ValueError, match=r"The circuit type requested is invalid."):
        test_job._get_circuits("invalid_type")


def test_compiled_circuits(backend: qss.SuperstaqBackend) -> None:
    response = mock_response("Queued")
    response["compiled_circuit"] = qss.serialize_circuits(qiskit.QuantumCircuit(2))
    response["input_circuit"] = qss.serialize_circuits(qiskit.QuantumCircuit(2))
    response["pulse_gate_circuits"] = None

    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with patched_requests({"123abc": response}) as mocked_get_job:
        assert job.compiled_circuits() == [qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == [qiskit.QuantumCircuit(2)]
    assert job.compiled_circuits(index=0) == qiskit.QuantumCircuit(2)

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456xyz")
    with patched_requests({"123abc": response, "456xyz": response}):
        assert job.compiled_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

        with pytest.raises(
            ValueError,
            match=r"The circuit type 'pulse_gate_circuits' is not supported on this device.",
        ):
            job.pulse_gate_circuits()

    assert job.compiled_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]


def test_compiled_circuitsV3(backendV3: qss.SuperstaqBackend) -> None:
    job_dict = job_dictV3()
    job_dict["statuses"] = ["awaiting_submission"]
    job_dict["compiled_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))]
    job_dict["input_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))]

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.compiled_circuits() == [qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == [qiskit.QuantumCircuit(2)]
    assert job.compiled_circuits(index=0) == qiskit.QuantumCircuit(2)

    job_dict = job_dictV3(2)
    job_dict["statuses"] = ["awaiting_submission"] * 2
    job_dict["compiled_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))] * 2
    job_dict["input_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))] * 2

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.compiled_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    assert job.compiled_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]

    job.job_info.compiled_circuits[0] = None
    with pytest.raises(gss.SuperstaqException, match=r"Some compiled circuits are missing"):
        job.compiled_circuits()

    with pytest.raises(gss.SuperstaqException, match=f"Circuit 0 of job {uuid.UUID(int=42)}"):
        job.compiled_circuits(index=0)

    job.job_info.compiled_circuits[1] = None
    with pytest.raises(
        gss.SuperstaqException, match=f"The job {uuid.UUID(int=42)} has no compiled circuits"
    ):
        job.compiled_circuits()


def test_index_compiled_circuits(backend: qss.SuperstaqBackend) -> None:
    response = mock_response("Done")
    single_qc = qiskit.QuantumCircuit(2, metadata={"test_label": "test_data"})
    qc_list = [single_qc, single_qc, single_qc]

    response["compiled_circuit"] = qss.serialize_circuits(single_qc)
    response["input_circuit"] = qss.serialize_circuits(single_qc)

    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with patched_requests({"123abc": response}) as mocked_get_job:
        assert job.compiled_circuits() == [single_qc]
        assert job.compiled_circuits()[0].metadata == {"test_label": "test_data"}
        assert job.compiled_circuits(index=0) == single_qc
        assert job.compiled_circuits(0).metadata == {"test_label": "test_data"}
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == [single_qc]
    assert job.compiled_circuits()[0].metadata == {"test_label": "test_data"}
    assert job.compiled_circuits(index=0) == single_qc
    assert job.compiled_circuits(0).metadata == {"test_label": "test_data"}

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456xyz,789cba")
    with patched_requests(
        {"123abc": response, "456xyz": response, "789cba": response}
    ) as mocked_get_job:
        assert job.compiled_circuits() == qc_list
        assert job.compiled_circuits(index=2) == single_qc
        for circ in job.compiled_circuits():
            assert circ.metadata == {"test_label": "test_data"}
        assert job.compiled_circuits(index=2).metadata == {"test_label": "test_data"}
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == qc_list
    assert job.compiled_circuits(index=2) == single_qc
    for circ in job.compiled_circuits():
        assert circ.metadata == {"test_label": "test_data"}
    assert job.compiled_circuits(index=2).metadata == {"test_label": "test_data"}


def test_index_compiled_circuitsV3(backendV3: qss.SuperstaqBackend) -> None:
    job_dict = job_dictV3()
    single_qc = qiskit.QuantumCircuit(2, metadata={"test_label": "test_data"})

    job_dict["compiled_circuits"] = [qss.serialize_circuits(single_qc)]
    job_dict["input_circuit"] = [qss.serialize_circuits(single_qc)]

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.compiled_circuits() == [single_qc]
        assert job.compiled_circuits()[0].metadata == {"test_label": "test_data"}
        assert job.compiled_circuits(index=0) == single_qc
        assert job.compiled_circuits(0).metadata == {"test_label": "test_data"}
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == [single_qc]
    assert job.compiled_circuits()[0].metadata == {"test_label": "test_data"}
    assert job.compiled_circuits(index=0) == single_qc
    assert job.compiled_circuits(0).metadata == {"test_label": "test_data"}

    job_dict = job_dictV3(3)
    job_dict["compiled_circuits"] = [qss.serialize_circuits(single_qc)] * 3
    job_dict["input_circuit"] = [qss.serialize_circuits(single_qc)] * 3
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.compiled_circuits() == [single_qc] * 3
        assert job.compiled_circuits(index=2) == single_qc
        for circ in job.compiled_circuits():
            assert circ.metadata == {"test_label": "test_data"}
        assert job.compiled_circuits(index=2).metadata == {"test_label": "test_data"}
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.compiled_circuits() == [single_qc] * 3
    assert job.compiled_circuits(index=2) == single_qc
    for circ in job.compiled_circuits():
        assert circ.metadata == {"test_label": "test_data"}
    assert job.compiled_circuits(index=2).metadata == {"test_label": "test_data"}


def test_input_circuits(backend: qss.SuperstaqBackend) -> None:
    response = mock_response("Queued")
    response["input_circuit"] = qss.serialize_circuits(qiskit.QuantumCircuit(2))

    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    with patched_requests({"123abc": response}) as mocked_get_job:
        assert job.input_circuits() == [qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.input_circuits() == [qiskit.QuantumCircuit(2)]
    assert job.input_circuits(index=0) == qiskit.QuantumCircuit(2)

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456xyz")
    with patched_requests({"123abc": response, "456xyz": response}):
        assert job.input_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    assert job.input_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
    assert job.input_circuits(index=1) == qiskit.QuantumCircuit(2)


def test_input_circuitsV3(backendV3: qss.SuperstaqBackend) -> None:
    job_dict = job_dictV3()
    job_dict["input_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))]

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.input_circuits() == [qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.input_circuits() == [qiskit.QuantumCircuit(2)]
    assert job.input_circuits(index=0) == qiskit.QuantumCircuit(2)

    job_dict = job_dictV3(2)
    job_dict["input_circuits"] = [qss.serialize_circuits(qiskit.QuantumCircuit(2))] * 2

    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with patched_requests({str(uuid.UUID(int=42)): job_dict}) as mocked_get_job:
        assert job.input_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
        mocked_get_job.assert_called_once()

    assert job.input_circuits() == [qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)]
    assert job.input_circuits(index=1) == qiskit.QuantumCircuit(2)


def test_pulse_gate_circuits(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    response = mock_response("Done")

    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    response["pulse_gate_circuits"] = qss.serialize_circuits(pulse_gate_circuit)

    with patched_requests({"123abc": response}) as mocked_get_job:
        assert job.pulse_gate_circuits()[0] == pulse_gate_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.pulse_gate_circuits()[0] == pulse_gate_circuit


def test_index_pulse_gate_circuits(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    response = mock_response("Done")

    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    response["pulse_gate_circuits"] = qss.serialize_circuits(pulse_gate_circuit)

    with patched_requests({"123abc": response}) as mocked_get_job:
        assert job.pulse_gate_circuits(index=0) == pulse_gate_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.pulse_gate_circuits(index=0) == pulse_gate_circuit

    # Test on invalid index
    with pytest.raises(ValueError, match=r"is less than the minimum"):
        job.pulse_gate_circuits(index=-3)


def test_multi_pulse_gate_circuits(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="123abc")
    response = mock_response("Done")
    pulse_gate_circuit = qiskit.QuantumCircuit(1, 1)
    pulse_gate = qiskit.circuit.Gate("test_pulse_gate", 1, [3.14, 1])
    pulse_gate_circuit.append(pulse_gate, [0])
    pulse_gate_circuit.measure(0, 0)

    response["pulse_gate_circuits"] = qss.serialize_circuits(pulse_gate_circuit)

    pgc_list = [
        pulse_gate_circuit,
        pulse_gate_circuit,
        pulse_gate_circuit,
    ]

    job = qss.SuperstaqJob(backend=backend, job_id="123abc,456xyz,789cba")
    with patched_requests({"123abc": response, "456xyz": response, "789cba": response}):
        assert job.pulse_gate_circuits() == pgc_list

    # After fetching the job info once it shouldn't be refreshed again (so no need to mock)
    assert job.pulse_gate_circuits() == pgc_list


def test_submit(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="12345")
    with pytest.raises(NotImplementedError, match=r"Submit through SuperstaqBackend"):
        job.submit()


def test_submitV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with pytest.raises(NotImplementedError, match=r"Submit through SuperstaqBackend"):
        job.submit()


def test_eq(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="12345")
    assert job != "super.tech"

    job2 = qss.SuperstaqJob(backend=backend, job_id="123456")
    assert job != job2

    job3 = qss.SuperstaqJob(backend=backend, job_id="12345")
    assert job == job3


def test_hash(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="12345")
    job2 = qss.SuperstaqJob(backend=backend, job_id="123456")
    job3 = qss.SuperstaqJob(backend=backend, job_id="12345")
    hash_set = set()
    hash_set.add(job)
    hash_set.add(job2)
    hash_set.add(job3)
    assert len(hash_set) == 2


def test_eqV3(backend: qss.SuperstaqBackend, backendV3: qss.SuperstaqBackend) -> None:
    jobV2 = qss.SuperstaqJob(backend, str(uuid.UUID(int=42)))
    job = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    assert job != jobV2

    job2 = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=43))
    assert job != job2

    job2 = qss.SuperstaqJobV3(backendV3, uuid.UUID(int=42))
    assert job == job2


def test_to_dict(backend: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJob(backend=backend, job_id="12345")
    with patched_requests({"12345": mock_response("Done")}):
        assert job.to_dict() == {
            "12345": {
                "status": "Done",
                "samples": {"11": 50, "10": 50},
                "shots": 100,
            }
        }


def test_to_dictV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job_dict = job_dictV3()
    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        result_dict = job.to_dict()
        result_dict["submission_timestamp"] = str(result_dict["submission_timestamp"])
        result_dict["last_updated_timestamp"] = list(
            map(str, result_dict["last_updated_timestamp"])
        )
        assert result_dict == job_dict


def test_metadataV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    job_dict = job_dictV3()
    with patched_requests({str(uuid.UUID(int=42)): job_dict}):
        assert job.metadata["foo"] == "bar"
        assert job.tags == ["some", "tags"]


def test_job_id(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    assert job.job_id() == uuid.UUID(int=42)


def test_hashV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    assert hash(job) == hash(uuid.UUID(int=42))


def test_job_infoV3(backendV3: qss.SuperstaqBackend) -> None:
    job = qss.SuperstaqJobV3(backend=backendV3, job_id=uuid.UUID(int=42))
    with (
        mock.patch.object(job, "_refresh_job", return_value=None),
        pytest.raises(AttributeError, match=r"Job info has not been fetched yet"),
    ):
        _ = job.job_info
