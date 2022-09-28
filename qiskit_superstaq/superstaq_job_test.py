import json
from typing import Any, Dict

import general_superstaq as gss
import pytest
import qiskit
import requests

import qiskit_superstaq as qss


class MockConfiguration:
    backend_name = "superstaq_backend"
    backend_version = gss.API_VERSION


class MockProvider(qss.SuperstaQProvider):
    def __init__(self) -> None:
        self.api_key = "very.tech"


class MockDevice(qss.SuperstaQBackend):
    def __init__(self) -> None:
        self._provider = MockProvider()
        self.diff = ""

    _configuration = MockConfiguration()

    remote_host = "super.tech"


class MockJob(qss.SuperstaQJob):
    def __init__(self) -> None:
        self._backend = MockDevice()
        self._job_id = "123abc"
        self.qobj = None


class MockJobs(qss.SuperstaQJob):
    def __init__(self) -> None:
        self._backend = MockDevice()
        self._job_id = "123abc,456def"
        self.qobj = None


class MockResponse:
    def __init__(self, status_str: str) -> None:
        self.content = json.dumps({"status": status_str, "samples": None, "shots": 100})

    def json(self) -> Dict:
        return json.loads(self.content)


def test_wait_for_results(monkeypatch: Any) -> None:

    job = MockJob()

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Done"))
    assert job._wait_for_results() == [{"status": "Done", "samples": None, "shots": 100}]

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Error"))

    with pytest.raises(qiskit.providers.JobError, match="API returned error"):
        job._wait_for_results()

    jobs = MockJobs()

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Done"))
    assert jobs._wait_for_results() == [
        {"status": "Done", "samples": None, "shots": 100},
        {"status": "Done", "samples": None, "shots": 100},
    ]


def test_result(monkeypatch: Any) -> None:
    job = MockJob()

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Done"))

    expected_results = [{"success": True, "shots": 100, "data": {"counts": None}}]

    expected = qiskit.result.Result.from_dict(
        {
            "results": expected_results,
            "qobj_id": -1,
            "backend_name": "superstaq_backend",
            "backend_version": gss.API_VERSION,
            "success": True,
            "job_id": "123abc",
        }
    )

    ans = job.result()

    assert ans.backend_name == expected.backend_name
    assert ans.job_id == expected.job_id


def test_status(monkeypatch: Any) -> None:
    job = MockJob()

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Queued"))
    assert job.status() == qiskit.providers.JobStatus.QUEUED

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Running"))
    assert job.status() == qiskit.providers.JobStatus.RUNNING

    monkeypatch.setattr(requests, "get", lambda *_, **__: MockResponse("Done"))
    assert job.status() == qiskit.providers.JobStatus.DONE


def test_submit() -> None:
    job = qss.SuperstaQJob(backend=MockDevice(), job_id="12345")
    with pytest.raises(NotImplementedError, match="Submit through SuperstaQBackend"):
        job.submit()


def test_eq() -> None:
    job = qss.SuperstaQJob(backend=MockDevice(), job_id="12345")
    assert job != "super.tech"

    job2 = qss.SuperstaQJob(backend=MockDevice(), job_id="123456")
    assert job != job2

    job3 = qss.SuperstaQJob(backend=MockDevice(), job_id="12345")
    assert job == job3
