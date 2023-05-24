# pylint: disable=missing-function-docstring,missing-class-docstring
from typing import Dict, Union
from unittest import mock

import general_superstaq as gss
import pytest
import qiskit

import qiskit_superstaq as qss


def mock_response(status_str: str) -> Dict[str, Union[str, int, Dict[str, int]]]:
    return {"status": status_str, "samples": {"10": 100}, "shots": 100}


@pytest.fixture
def backend() -> qss.SuperstaQBackend:
    provider = qss.SuperstaQProvider(api_key="token")
    return provider.get_backend("ss_example_qpu")


def test_job_id(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")
    assert job.get_job_id() == "123abc"


def test_get_backend(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")
    assert job.get_backend() == backend


def test_wait_for_results(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")
    jobs = qss.SuperstaQJob(backend=backend, job_id="123abc,456def")

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done"),
    ):
        assert job._wait_for_results() == [mock_response("Done")]
        assert jobs._wait_for_results() == [mock_response("Done"), mock_response("Done")]

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Error"),
    ):
        with pytest.raises(qiskit.providers.JobError, match="API returned error"):
            _ = job._wait_for_results()


def test_timeout(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    with pytest.raises(qiskit.providers.JobTimeoutError, match="Timed out waiting for result"):
        _ = job._wait_for_results(timeout=-1.0)

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        side_effect=[mock_response("Queued"), mock_response("Queued"), mock_response("Done")],
    ) as mocked_get_job:
        assert job._wait_for_results(wait=0.0) == [mock_response("Done")]
        assert mocked_get_job.call_count == 3


def test_result(backend: qss.SuperstaQBackend) -> None:
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
        return_value=mock_response("Done"),
    ):
        ans = job.result()

        assert ans.backend_name == expected.backend_name
        assert ans.job_id == expected.job_id


def test_check_if_stopped(backend: qss.SuperstaQBackend) -> None:

    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    for status in job.STOPPED_STATES:
        job._job_info["status"] = status
        with pytest.raises(gss.SuperstaQUnsuccessfulJobException, match=status):
            _ = job.get_backend()


def test_refresh_job(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="123abc,456abc,789abc")

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Queued"),
    ):
        job._refresh_job()
        assert job._job_info["status"] == "Queued"

    for status_msg in job.STOPPED_STATES:
        with mock.patch(
            "general_superstaq.superstaq_client._SuperstaQClient.get_job",
            return_value=mock_response(status_msg),
        ):
            job._refresh_job()
            assert job._job_info["status"] == "Canceled"

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Running"),
    ):
        job._refresh_job()
        assert job._job_info["status"] == "Running"

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done"),
    ):
        job._refresh_job()
        assert job._job_info["status"] == "Done"


def test_status(backend: qss.SuperstaQBackend) -> None:

    job = qss.SuperstaQJob(backend=backend, job_id="123abc")

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Queued"),
    ):
        assert job.status() == qiskit.providers.JobStatus.QUEUED

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Running"),
    ):
        assert job.status() == qiskit.providers.JobStatus.RUNNING

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaQClient.get_job",
        return_value=mock_response("Done"),
    ):
        assert job.status() == qiskit.providers.JobStatus.DONE

    job = qss.SuperstaQJob(backend=backend, job_id="123done")
    for status_msg in job.TERMINAL_STATES:
        if status_msg == "Done":
            job._job_info["status"] = "Done"
            assert job.status() == qiskit.providers.JobStatus.DONE
        else:
            job._job_info["status"] = "Canceled"
            assert job.status() == qiskit.providers.JobStatus.CANCELLED


def test_submit(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="12345")
    with pytest.raises(NotImplementedError, match="Submit through SuperstaQBackend"):
        job.submit()


def test_eq(backend: qss.SuperstaQBackend) -> None:
    job = qss.SuperstaQJob(backend=backend, job_id="12345")
    assert job != "super.tech"

    job2 = qss.SuperstaQJob(backend=backend, job_id="123456")
    assert job != job2

    job3 = qss.SuperstaQJob(backend=backend, job_id="12345")
    assert job == job3
