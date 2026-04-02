# Copyright 2026 Infleqtion
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
import textwrap
import uuid
from unittest import mock

import pytest
import requests

import general_superstaq as gss


@pytest.fixture
def mock_client() -> gss.superstaq_client._SuperstaqClientV3:
    """Fixture for general-superstaq client."""
    return gss.superstaq_client._SuperstaqClientV3(
        client_name="general-superstaq",
        remote_host="http://example.com",
        api_key="to_my_heart",
        api_version="v0.3.0",
        circuit_type=gss.models.CircuitType.QASM,
    )


def _job_dict() -> dict[str, object]:
    """Fixture for a standard, completed single v0.3.0 job result.

    Returns:
        A dictionary containing commonly expected job data.
    """
    return {
        "job_type": "simulate",
        "statuses": ["completed"],
        "status_messages": [None],
        "user_email": "test@email.com",
        "target": "ss_unconstrained_simulator",
        "provider_id": ["provider_id"],
        "num_circuits": 1,
        "compiled_circuits": [None],
        "input_circuits": [""],
        "circuit_type": "qasm",
        "counts": [{"11": 1}],
        "results_dicts": [],
        "shots": [1],
        "dry_run": True,
        "submission_timestamp": str(datetime.datetime.now(tz=datetime.timezone.utc)),
        "last_updated_timestamp": [str(datetime.datetime.now(tz=datetime.timezone.utc))],
        "initial_logical_to_physicals": [{0: 0, 1: 1}],
        "final_logical_to_physicals": [{0: 0, 1: 1}],
        "logical_qubits": ["0", "1"],
        "physical_qubits": ["0", "1"],
        "tags": ["some", "tags"],
        "metadata": {"foo": "bar"},
    }


def _mocked_response(content: object) -> requests.Response:
    response = requests.Response()
    response.status_code = requests.codes.OK
    response._content = json.dumps(content).encode()
    return response


@mock.patch("requests.Session.put")
def test_cancel(
    mock_put: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    mock_put.return_value = _mocked_response({"succeeded": ["circuit"], "message": "message"})

    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    job.cancel()
    mock_put.assert_called_once()


@mock.patch("requests.Session.get")
def test_fields(
    mock_get: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, uuid.UUID(int=123))

    assert job.job_id() == uuid.UUID(int=123)
    mock_get.assert_not_called()

    assert job.target() == "ss_unconstrained_simulator"
    assert job._repetitions() == 1
    assert job.metadata["foo"] == "bar"
    assert job.tags == ["some", "tags"]
    mock_get.assert_called_once()

    # Shouldn't need to retrieve anything now that `job._job_data` is populated:
    assert job.target() == "ss_unconstrained_simulator"
    assert job._repetitions() == 1
    assert job.metadata["foo"] == "bar"
    assert job.tags == ["some", "tags"]
    mock_get.assert_called_once()


@mock.patch("requests.Session.get")
def test_to_dict(
    mock_get: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    assert job.to_dict() == gss.models.JobData(**job_dict).model_dump()


def test_equality(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job1 = gss.job.Job(mock_client, uuid.UUID(int=123))
    job2 = gss.job.Job(mock_client, uuid.UUID(int=123))
    job3 = gss.job.Job(mock_client, uuid.UUID(int=456))

    assert job1 == job2
    assert job1 != job3
    assert job2 != job3
    assert job1 != 123

    assert {job1, job2, job3} == {job1, job3}


@mock.patch("requests.Session.get")
def test_refresh_job(
    mock_get: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:

    job_dict = _job_dict()
    job_dict["num_circuits"] = 3
    job_dict["statuses"] = ["running", "awaiting_submission", "completed"]
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, job_id=uuid.UUID(int=123))
    assert job._overall_status == "received"

    job._refresh_job()
    assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["running", "completed", "completed"]
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job._refresh_job()
    assert job._overall_status == "running"

    job_dict["statuses"] = ["completed", "completed", "completed"]
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job._refresh_job()
    assert job._overall_status == "completed"

    job_dict["statuses"] = ["failed"]
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job._refresh_job()
    assert job._overall_status == "completed"  # No update because already in terminal state

    job = gss.job.Job(mock_client, job_id=uuid.UUID(int=123))
    assert job._overall_status == "received"

    job._refresh_job()
    assert job._overall_status == "failed"

    job_dict["statuses"] = ["cancelled"]
    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, job_id=uuid.UUID(int=123))
    assert job._overall_status == "received"

    job._refresh_job()
    assert job._overall_status == "cancelled"


def test_status_refresh(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job_dict = _job_dict()
    mock_complete_response = _mocked_response({str(uuid.UUID(int=123)): job_dict.copy()})

    for status in gss.models.CircuitStatus:
        if status not in gss.models.TERMINAL_CIRCUIT_STATES:
            mock_incomplete_response = _mocked_response(
                {str(uuid.UUID(int=123)): {**job_dict, "statuses": [status.value]}}
            )

            job = gss.job.Job(mock_client, uuid.UUID(int=123))
            with mock.patch(
                "requests.Session.get",
                side_effect=[mock_incomplete_response, mock_complete_response],
            ) as mock_get:
                assert job._status() == status
                assert job._status() == "completed"
                assert mock_get.call_count == 2
                assert (
                    mock_get.call_args[0][0]
                    == f"http://example.com/v0.3.0/client/job/qasm?job_id={uuid.UUID(int=123)}"
                )

    for status in gss.models.TERMINAL_CIRCUIT_STATES:
        job = gss.job.Job(mock_client, uuid.UUID(int=123))
        with mock.patch("requests.Session.get") as mock_get:
            mock_get.return_value = _mocked_response(
                {str(uuid.UUID(int=123)): {**job_dict, "statuses": [status.value]}}
            )
            assert job._status() == status
            assert job._status() == status
            mock_get.assert_called_once()
            assert (
                mock_get.call_args[0][0]
                == f"http://example.com/v0.3.0/client/job/qasm?job_id={uuid.UUID(int=123)}"
            )


@mock.patch("time.sleep", return_value=None)
def test_wait_until_completed_poll(
    mock_sleep: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()

    running_mock = _mocked_response(
        {str(uuid.UUID(int=123)): {**job_dict, "statuses": ["running"]}}
    )
    completed_mock = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    with mock.patch("requests.Session.get", side_effect=[running_mock, completed_mock]) as mock_get:
        job.wait_until_complete(index=0, polling_seconds=0)
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    assert job.job_data.statuses == ["completed"]


@mock.patch("time.sleep", return_value=None)
def test_wait_until_completed_poll_timeout(
    mock_sleep: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()
    job_dict["statuses"] = ["running"]

    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    with mock.patch("requests.Session.get") as mock_get:
        mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})
        with pytest.raises(
            TimeoutError, match=r"Timed out while waiting for results. Final status was 'running'"
        ):
            job.wait_until_complete(index=0, timeout_seconds=5, polling_seconds=10)

        mock_sleep.assert_called_once()
        assert mock_get.call_count == 2


@mock.patch("requests.Session.get")
def test_multijob_overall_status(
    mock_get: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()
    job_dict["num_circuits"] = 4
    job_dict["statuses"] = ["running", "pending", "pending", "completed"]

    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    # Test "running" status dominates as the overall status
    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    assert job._status() == "running"
    assert job._status(index=2) == "pending"
    assert job._status(index=3) == "completed"


def test_update_status_queue_info(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job = gss.job.Job(mock_client, job_id=uuid.UUID(int=42))

    job_dict = _job_dict()
    job_dict["num_circuits"] = 3
    job_dict["statuses"] = ["completed"] * 3
    job._job_data = gss.models.JobData(**job_dict)

    job._update_status_queue_info()
    assert job._overall_status == "completed"

    job_dict["statuses"] = ["awaiting_submission", "cancelled", "cancelled"]
    job._job_data = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["cancelled", "cancelled", "awaiting_submission"]
    job._job_data = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "awaiting_submission"

    job_dict["statuses"] = ["completed", "completed", "failed"]
    job._job_data = gss.models.JobData(**job_dict)
    job._update_status_queue_info()
    assert job._overall_status == "failed"


@mock.patch("requests.Session.get")
def test_check_if_unsuccessful(
    mock_get: mock.MagicMock, mock_client: gss.superstaq_client._SuperstaqClientV3
) -> None:
    job_dict = _job_dict()
    job_dict["num_circuits"] = 2
    job_dict["statuses"] = ["completed", "failed"]
    job_dict["status_messages"] = [None, "failure"]

    mock_get.return_value = _mocked_response({str(uuid.UUID(int=123)): job_dict})

    job = gss.job.Job(mock_client, uuid.UUID(int=123))

    with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="failure"):
        job._check_if_unsuccessful()

    job._check_if_unsuccessful(0)
    with pytest.raises(gss.SuperstaqUnsuccessfulJobException, match="failure"):
        job._check_if_unsuccessful(1)


def test_str(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    assert str(job) == f"Job with job_id={uuid.UUID(int=123)}"


def test_getitem(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    with pytest.raises(NotImplementedError):
        job.__getitem__(0)
    with pytest.raises(NotImplementedError):
        job[0]


def test_job_data_failure(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job = gss.job.Job(mock_client, uuid.UUID(int=123))
    with (
        mock.patch.object(job, "_refresh_job", return_value=None),
        pytest.raises(AttributeError, match=r"Job data has not been fetched yet"),
    ):
        _ = job.job_data


def test_set_counts(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    job = gss.job.Job(mock_client, uuid.UUID(int=123))

    job_dict = _job_dict()
    job_dict["num_circuits"] = 2
    job_dict["counts"] = [None, None]
    job_dict["final_logical_to_physicals"] = 2 * [{0: 0, 1: 1, 2: 2}]
    job._job_data = gss.models.JobData(**job_dict)

    counts = [{"001": 1, "010": 10, "100": 100}, {"011": 11, "101": 101, "110": 110}]

    job.set_counts(counts)
    assert job.job_data.counts == counts

    job.set_counts_for_circuit(1, counts[0])
    job.set_counts_for_circuit(0, counts[1])
    assert job.job_data.counts == counts[::-1]


def test_set_counts_jaqal(mock_client: gss.superstaq_client._SuperstaqClientV3) -> None:
    jaqalpaq_run = pytest.importorskip("jaqalpaq.run")

    job = gss.job.Job(mock_client, uuid.UUID(int=123))

    job_dict = _job_dict()
    job_dict["num_circuits"] = 3
    job_dict["counts"] = [None, None, None]
    job_dict["final_logical_to_physicals"] = 3 * [{0: 0, 1: 1, 2: 2}]
    job._job_data = gss.models.JobData(**job_dict)

    jaqal_str = textwrap.dedent(
        """\
        from qscout.v1.std usepulses *

        register allqubits[3]

        prepare_all
        R allqubits[0] 0 3.141592653589793
        measure_all

        prepare_all
        R allqubits[1] 0 1.5708
        R allqubits[2] 0 3.141592653589793
        measure_all

        prepare_all
        MS allqubits[0] allqubits[1] 0 1.5708
        measure_all
        """
    )
    result = jaqalpaq_run.run_jaqal_string(jaqal_str, overrides={"__repeats__": 1000})

    job.set_counts(result)
    assert job.job_data.counts[0]
    assert job.job_data.counts[1]
    assert job.job_data.counts[2]

    assert job.job_data.counts[0] == {"100": 1000}
    assert job.job_data.counts[1].keys() == {"001", "011"}
    assert job.job_data.counts[2].keys() == {"000", "110"}

    job.set_counts_for_circuit(2, result.by_subbatch[0].by_subcircuit[0])
    job.set_counts_for_circuit(0, result.by_subbatch[0].by_subcircuit[1])
    job.set_counts_for_circuit(1, result.by_subbatch[0].by_subcircuit[2])
    assert job.job_data.counts[0].keys() == {"001", "011"}
    assert job.job_data.counts[1].keys() == {"000", "110"}
    assert job.job_data.counts[2] == {"100": 1000}

    job.job_data.final_logical_to_physicals[0] = {0: 1, 1: 2, 2: 0}
    job.job_data.final_logical_to_physicals[1] = {0: 2, 1: 0, 2: 1}
    job.job_data.final_logical_to_physicals[2] = {0: 0, 1: 2, 2: 1}
    job.set_counts(result.by_subbatch[0])
    assert job.job_data.counts[0] == {"001": 1000}
    assert job.job_data.counts[1].keys() == {"100", "101"}
    assert job.job_data.counts[2].keys() == {"000", "101"}

    with pytest.raises(ValueError, match="must be compiled"):
        job.job_data.final_logical_to_physicals[0] = None
        job.set_counts(result)
