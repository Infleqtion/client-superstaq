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

import applications_superstaq
import cirq
import pytest

import cirq_superstaq


def test_job_fields() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "job_id": "my_id",
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
    job = cirq_superstaq.Job(mock_client, job_dict)
    assert job.job_id() == "my_id"
    assert job.target() == "simulator"
    assert job.num_qubits() == 2
    assert job.repetitions() == 1


def test_job_status_refresh() -> None:
    for status in cirq_superstaq.Job.NON_TERMINAL_STATES:
        mock_client = mock.MagicMock()
        mock_client.get_job.return_value = {"job_id": "my_id", "status": "completed"}
        job = cirq_superstaq.Job(mock_client, {"job_id": "my_id", "status": status})
        assert job.status() == "completed"
        mock_client.get_job.assert_called_with("my_id")
    for status in cirq_superstaq.Job.TERMINAL_STATES:
        mock_client = mock.MagicMock()
        job = cirq_superstaq.Job(mock_client, {"job_id": "my_id", "status": status})
        assert job.status() == status
        mock_client.get_job.assert_not_called()


def test_job_str_repr_eq() -> None:
    client = applications_superstaq.superstaq_client._SuperstaQClient(
        client_name="applications-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    job = cirq_superstaq.Job(client, {"job_id": "my_id"})
    assert str(job) == "Job with job_id=my_id"
    cirq.testing.assert_equivalent_repr(
        job, setup_code="import cirq_superstaq\nimport applications_superstaq"
    )

    assert not job == 1


def test_job_counts() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "job_id": "my_id",
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
    job = cirq_superstaq.Job(mock_client, job_dict)
    results = job.counts()
    assert results == {"11": 1}


def test_job_counts_failed() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "job_id": "my_id",
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
    job = cirq_superstaq.Job(mock_client, job_dict)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.counts()
    assert job.status() == "Failed"


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll(mock_sleep: mock.MagicMock) -> None:
    ready_job = {
        "job_id": "my_id",
        "status": "ready",
    }
    completed_job = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "job_id": "my_id",
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
    job = cirq_superstaq.Job(mock_client, ready_job)
    results = job.counts(polling_seconds=0)
    assert results == {"11": 1}
    mock_sleep.assert_called_once()


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll_timeout(mock_sleep: mock.MagicMock) -> None:
    ready_job = {
        "job_id": "my_id",
        "status": "ready",
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = cirq_superstaq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match="ready"):
        _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_timeout_with_error_message(mock_sleep: mock.MagicMock) -> None:
    ready_job = {"job_id": "my_id", "status": "failure", "failure": {"error": "too many qubits"}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = cirq_superstaq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


def test_job_fields_unsuccessful() -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "job_id": "my_id",
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
    job = cirq_superstaq.Job(mock_client, job_dict)
    with pytest.raises(applications_superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.target()
    with pytest.raises(applications_superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.num_qubits()
    with pytest.raises(applications_superstaq.SuperstaQUnsuccessfulJobException, match="Deleted"):
        _ = job.repetitions()
