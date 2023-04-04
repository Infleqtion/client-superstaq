# pylint: disable=missing-function-docstring,missing-class-docstring
from typing import Dict, Union

import pytest
import qiskit

import qiskit_superstaq as qss


def mock_response(status_str: str) -> Dict[str, Union[str, int, Dict[str, int]]]:
    return {"status": status_str, "samples": {"10": 100}, "shots": 100}


def test_wait_for_results() -> None:
    backend = qss.SuperstaQProvider(api_key="token").get_backend("ss_example_qpu")
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done"),
    ):
        assert job._wait_for_results() == [mock_response("Done")]

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Error"),
    ):
        with pytest.raises(qiskit.providers.JobError, match="API returned error"):
            _ = job._wait_for_results()

    jobs = qss.SuperstaQJob(backend=backend, job_id="123abc,456def")
    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done"),
    ):
        assert jobs._wait_for_results() == [mock_response("Done"), mock_response("Done")]


def test_result() -> None:
    backend = qss.SuperstaQProvider(api_key="token").get_backend("ss_example_qpu")
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    expected_results = [{"success": True, "shots": 100, "data": {"counts": {"01": 100}}}]

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

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done")
    ):
        ans = job.result()

        assert ans.backend_name == expected.backend_name
        assert ans.job_id == expected.job_id


def test_status() -> None:
    backend = qss.SuperstaQProvider(api_key="token").get_backend("ss_example_qpu")
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Queued")
    ):
        assert job.status() == qiskit.providers.JobStatus.QUEUED

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Running")
    ):
        assert job.status() == qiskit.providers.JobStatus.RUNNING

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done")
    ):
        assert job.status() == qiskit.providers.JobStatus.DONE


def test_submit() -> None:
    backend = qss.SuperstaQProvider(api_key="token").get_backend("ss_example_qpu")
    job = qss.SuperstaQJob(backend=backend, job_id="12345")
    with pytest.raises(NotImplementedError, match="Submit through SuperstaQBackend"):
        job.submit()


def test_eq() -> None:
    backend = qss.SuperstaQProvider(api_key="token").get_backend("ss_example_qpu")

    job = qss.SuperstaQJob(backend=backend, job_id="12345")
    assert job != "super.tech"

    job2 = qss.SuperstaQJob(backend=backend, job_id="123456")
    assert job != job2

    job3 = qss.SuperstaQJob(backend=backend, job_id="12345")
    assert job == job3
