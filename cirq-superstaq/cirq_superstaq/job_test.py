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
# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

from typing import Any, Dict
from unittest import mock

import cirq
import general_superstaq as gss
import pytest

import cirq_superstaq as css


@pytest.fixture
def job() -> css.Job:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="cirq-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    return css.Job(client, "job_id")


def new_job() -> css.Job:
    client = gss.superstaq_client._SuperstaqClient(
        client_name="cirq-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
    )
    return css.Job(client, "new_job_id")


def mocked_get_job_requests(*job_dicts: Dict[str, Any]) -> mock._patch[mock.Mock]:
    """Mocks the server's response to `get_job` requests using the given sequence of job_dicts.
    Return type is wrapped in a string because "'type' object is not subscriptable"
    is thrown at runtime
    """
    return mock.patch(
        "general_superstaq.superstaq_client._SuperstaqClient.get_job", side_effect=job_dicts
    )


def test_job_fields(job: css.job.Job) -> None:
    compiled_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "samples": {"11": 1},
        "shots": 1,
        "status": "Done",
        "target": "ss_unconstrained_simulator",
        "compiled_circuit": css.serialize_circuits(compiled_circuit),
    }

    assert job.job_id() == "job_id"

    with mocked_get_job_requests(job_dict) as mocked_get_job:
        assert job.target() == "ss_unconstrained_simulator"
        assert job.num_qubits(index=0) == 2
        assert job.repetitions() == 1
        assert job.compiled_circuits(index=0) == compiled_circuit
        mocked_get_job.assert_called_once()  # Only refreshed once


def test_target(job: css.job.Job) -> None:
    job_dict = {"status": "Done", "target": "ss_unconstrained_simulator"}

    # The first call will trigger a refresh:
    with mocked_get_job_requests(job_dict) as mocked_get_job:
        assert job.target() == "ss_unconstrained_simulator"
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.target() == "ss_unconstrained_simulator"


def test_num_qubits(job: css.job.Job) -> None:
    job_dict = {"status": "Done", "num_qubits": 2}

    # The first call will trigger a refresh:
    with mocked_get_job_requests(job_dict) as mocked_get_job:
        # Test case: index -> int
        assert job.num_qubits(index=0) == 2
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.num_qubits(index=0) == 2

    # Deprecation warning test
    with pytest.warns(DeprecationWarning, match="the numbers of qubits in all circuits"):
        assert job.num_qubits() == 2


def test_repetitions(job: css.job.Job) -> None:
    job_dict = {"status": "Done", "shots": 1}

    # The first call will trigger a refresh:
    with mocked_get_job_requests(job_dict) as mocked_get_job:
        assert job.repetitions() == 1
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.repetitions() == 1


def test_get_circuit(job: css.job.Job) -> None:
    with pytest.raises(ValueError, match="The circuit type requested is invalid."):
        job._get_circuits("invalid_type")


def test_compiled_circuit(job: css.job.Job) -> None:
    compiled_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_dict = {"status": "Done", "compiled_circuit": css.serialize_circuits(compiled_circuit)}

    # The first call will trigger a refresh:
    with mocked_get_job_requests(job_dict) as mocked_get_job:
        assert job.compiled_circuits(index=0) == compiled_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.compiled_circuits(index=0) == compiled_circuit


def test_multi_circuit_job() -> None:
    job = css.Job(
        gss.superstaq_client._SuperstaqClient(
            client_name="cirq-superstaq",
            remote_host="http://example.com",
            api_key="to_my_heart",
        ),
        "abc123,def456,ghi789",
    )

    compiled_circuit = cirq.Circuit(
        cirq.H(cirq.q(0)),
        cirq.CNOT(cirq.q(2), cirq.q(0)),
        cirq.X(cirq.q(1)) ** 0.5,
        cirq.measure(cirq.q(0), cirq.q(1), cirq.q(2)),
    )
    job_dict = {
        "status": "Done",
        "num_qubits": 3,
        "data": {"histogram": {"000": 0.16, "010": 0.36, "100": 0.3, "110": 0.18}},
        "samples": {"000": 8, "010": 18, "100": 15, "110": 9},
        "shots": 50,
        "compiled_circuit": css.serialize_circuits(compiled_circuit),
        "input_circuit": css.serialize_circuits(compiled_circuit),
    }

    with mock.patch(
        "general_superstaq.superstaq_client._SuperstaqClient.get_job", return_value=job_dict
    ):
        # Test case: No index
        assert job.compiled_circuits() == [compiled_circuit, compiled_circuit, compiled_circuit]
        assert job.num_qubits(index=None) == [3, 3, 3]
        assert job.counts() == [
            {"000": 8, "010": 18, "100": 15, "110": 9},
            {"000": 8, "010": 18, "100": 15, "110": 9},
            {"000": 8, "010": 18, "100": 15, "110": 9},
        ]
        assert job.counts(qubit_indices=[0]) == [
            {"0": 26, "1": 24},
            {"0": 26, "1": 24},
            {"0": 26, "1": 24},
        ]

        # Test case: index
        assert job.compiled_circuits(index=2) == compiled_circuit
        assert job.num_qubits(index=2) == 3
        assert job.counts(index=2) == {"000": 8, "010": 18, "100": 15, "110": 9}
        assert job.counts(index=2, qubit_indices=[0]) == {"0": 26, "1": 24}


def test_input_circuit(job: css.job.Job) -> None:
    input_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.measure(cirq.q(0)))
    job_dict = {
        "status": "Done",
        "input_circuit": css.serialize_circuits(input_circuit),
        "compiled_circuit": css.serialize_circuits(input_circuit),
    }

    # The first call will trigger a refresh:
    with mocked_get_job_requests(job_dict) as mocked_get_job:
        assert job.input_circuits(index=0) == input_circuit
        mocked_get_job.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job` is populated:
    assert job.input_circuits(index=0) == input_circuit


def test_job_status_refresh() -> None:
    completed_job_dict = {"status": "Done"}

    for status in css.Job.NON_TERMINAL_STATES:
        job_dict = {"status": status}

        with mocked_get_job_requests(job_dict, completed_job_dict) as mocked_request:
            job = new_job()
            assert job.status() == status
            assert job.status() == "Done"
            assert mocked_request.call_count == 2
            mocked_request.assert_called_with("new_job_id")

    for status in css.Job.TERMINAL_STATES:
        job_dict = {"status": status}

        with mocked_get_job_requests(job_dict, completed_job_dict) as mocked_request:
            job = new_job()
            assert job.status() == status
            assert job.status() == status
            mocked_request.assert_called_once_with("new_job_id")


def test_value_equality(job: css.job.Job) -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(job, job)
    eq.add_equality_group(new_job())


def test_job_str_repr_eq(job: css.job.Job) -> None:
    assert str(job) == "Job with job_id=job_id"
    cirq.testing.assert_equivalent_repr(
        job, setup_code="import cirq_superstaq as css\nimport general_superstaq as gss"
    )

    assert not job == 1


def test_job_to_dict(job: css.job.Job) -> None:
    job_result = {
        "job_id": {
            "data": {"histogram": {"11": 1}},
            "num_qubits": 2,
            "samples": {"11": 1},
            "shots": 1,
            "status": "Done",
            "target": "ss_unconstrained_simulator",
        }
    }
    job._job = {}
    job_dict = {"job_id": job_result}
    with mocked_get_job_requests(job_result):
        assert job.to_dict() == job_dict


def test_job_counts(job: css.job.Job) -> None:
    job_dict = {
        "data": {"histogram": {"10": 1}},
        "num_qubits": 2,
        "samples": {"10": 1},
        "shots": 1,
        "status": "Done",
        "target": "ss_unconstrained_simulator",
    }
    with mocked_get_job_requests(job_dict):
        assert job.counts(index=0) == {"10": 1}
        assert job.counts(index=0, qubit_indices=[0]) == ({"1": 1})

    # Deprecation warning test
    with pytest.warns(DeprecationWarning, match="the counts in all circuits in this"):
        assert job.counts() == {"10": 1}


def test_job_counts_failed(job: css.job.Job) -> None:
    job_dict = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "samples": {"11": 1},
        "shots": 1,
        "status": "Failed",
        "failure": {"error": "too many qubits"},
        "target": "ss_unconstrained_simulator",
    }
    with mocked_get_job_requests(job_dict):
        with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="too many qubits"):
            _ = job.counts()
        assert job.status() == "Failed"


@mock.patch("time.sleep", return_value=None)
def test_job_counts_poll(mock_sleep: mock.MagicMock, job: css.job.Job) -> None:
    ready_job = {
        "status": "Ready",
    }
    completed_job = {
        "data": {"histogram": {"11": 1}},
        "num_qubits": 2,
        "samples": {"11": 1},
        "shots": 1,
        "status": "Done",
        "target": "ss_unconstrained_simulator",
    }

    with mocked_get_job_requests(ready_job, completed_job) as mocked_requests:
        results = job.counts(index=0, polling_seconds=0)
        assert results == {"11": 1}
        assert mocked_requests.call_count == 2
        mock_sleep.assert_called_once()


@mock.patch("time.sleep", return_value=None)
@mock.patch("time.time", side_effect=range(20))
def test_job_counts_poll_timeout(
    mock_time: mock.MagicMock, mock_sleep: mock.MagicMock, job: css.job.Job
) -> None:
    ready_job = {
        "status": "Ready",
    }
    with mocked_get_job_requests(*[ready_job] * 20):
        with pytest.raises(TimeoutError, match="Ready"):
            _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_failure(mock_sleep: mock.MagicMock, job: css.job.Job) -> None:
    running_job = {
        "status": "Running",
    }
    failed_job = {
        "status": "Failed",
        "failure": {"error": "too many qubits"},
    }

    with mocked_get_job_requests(*[running_job] * 5, failed_job):
        with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="too many qubits"):
            _ = job.counts(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 5


def test_get_marginal_counts() -> None:
    counts_dict = {"10": 50, "11": 50}
    indices = [0]
    assert css.job._get_marginal_counts(counts_dict, indices) == ({"1": 100})
