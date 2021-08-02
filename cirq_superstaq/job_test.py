# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import pytest

import cirq_superstaq as superstaq


def test_job_fields() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "simulator",
    }

    mock_client = mock.MagicMock()
    job = superstaq.Job(mock_client, job_dict)
    assert job.job_id() == "my_id"
    assert job.target() == "simulator"
    assert job.num_qubits() == 2
    assert job.repetitions() == 1


def test_job_status_refresh() -> None:
    for status in superstaq.Job.NON_TERMINAL_STATES:
        mock_client = mock.MagicMock()
        mock_client.get_job.return_value = {"id": "my_id", "status": "completed"}
        job = superstaq.Job(mock_client, {"id": "my_id", "status": status})
        assert job.status() == "completed"
        mock_client.get_job.assert_called_with("my_id")
    for status in superstaq.Job.TERMINAL_STATES:
        mock_client = mock.MagicMock()
        job = superstaq.Job(mock_client, {"id": "my_id", "status": status})
        assert job.status() == status
        mock_client.get_job.assert_not_called()


def test_job_str() -> None:
    mock_client = mock.MagicMock()
    job = superstaq.Job(mock_client, {"id": "my_id"})
    assert str(job) == "cirq_superstaq.Job(job_id=my_id)"


def test_job_counts() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "simulator",
    }
    mock_client = mock.MagicMock()
    job = superstaq.Job(mock_client, job_dict)
    results = job.counts()
    assert results == {"11": 1}


def test_job_counts_failed() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Failed",
        "failure": {"error": "too many qubits"},
        "target": "simulator",
    }
    mock_client = mock.MagicMock()
    job = superstaq.Job(mock_client, job_dict)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.counts()
    assert job.status() == "Failed"


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll(mock_sleep: mock.MagicMock) -> None:
    ready_job = {
        "id": "my_id",
        "status": "ready",
    }
    completed_job = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "simulator",
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = [ready_job, completed_job]
    job = superstaq.Job(mock_client, ready_job)
    results = job.counts(polling_seconds=0)
    assert results == {"11": 1}
    mock_sleep.assert_called_once()


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll_timeout(mock_sleep: mock.MagicMock) -> None:
    ready_job = {
        "id": "my_id",
        "status": "ready",
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = superstaq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match="ready"):
        _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_timeout_with_error_message(mock_sleep: mock.MagicMock) -> None:
    ready_job = {"id": "my_id", "status": "failure", "failure": {"error": "too many qubits"}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = superstaq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


def test_job_fields_unsuccessful() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "id": "my_id",
        "num_qubits": 2,
        "samples": {"11": 1},
        "shots": [
            {
                "shots": 1,
                "status": "Deleted",
            }
        ],
        "status": "Deleted",
        "target": "simulator",
    }
    mock_client = mock.MagicMock()
    job = superstaq.Job(mock_client, job_dict)
    with pytest.raises(superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.target()
    with pytest.raises(superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.num_qubits()
    with pytest.raises(superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.repetitions()
