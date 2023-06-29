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
"""Represents a job created via the Superstaq API."""
import time
from typing import Any, Dict, Tuple

import cirq
import general_superstaq as gss
from cirq._doc import document


@cirq.value_equality(unhashable=True)
class Job:
    """A job created on the Superstaq API.

    Note that this is mutable, when calls to get status or results are made the job updates itself
    to the results returned from the API.

    If a job is canceled or deleted, only the job id and the status remain valid.
    """

    TERMINAL_STATES = ("Done", "Canceled", "Failed", "Deleted")
    document(
        TERMINAL_STATES,
        "States of the Superstaq API job from which the job cannot transition. "
        "Note that deleted can only exist in a return call from a delete "
        "(subsequent calls will return not found).",
    )

    NON_TERMINAL_STATES = ("Ready", "Submitted", "Running")
    document(
        NON_TERMINAL_STATES, "States of the Superstaq API job which can transition to other states."
    )

    ALL_STATES = TERMINAL_STATES + NON_TERMINAL_STATES
    document(ALL_STATES, "All states that an Superstaq API job can exist in.")

    UNSUCCESSFUL_STATES = ("Canceled", "Failed", "Deleted")
    document(
        UNSUCCESSFUL_STATES,
        "States of the Superstaq API job when it was not successful and so does not have any "
        "data associated with it beyond an id and a status.",
    )

    def __init__(self, client: gss.superstaq_client._SuperstaqClient, job_id: str) -> None:
        """Construct a Job.

        Users should not call this themselves. If you only know the `job_id`, use `get_job`
        on `css.Service`.

        Args:
            client: The client used for calling the API.
            job_id: Unique identifier for the job.
        """
        self._client = client
        self._job: Dict[str, Any] = {"status": "Submitted"}
        self._job_id = job_id

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""
        if self._job["status"] not in self.TERMINAL_STATES:
            self._job = self._client.get_job(self.job_id())

    def _check_if_unsuccessful(self) -> None:
        status = self.status()
        if status in self.UNSUCCESSFUL_STATES:
            if "failure" in self._job and "error" in self._job["failure"]:
                # if possible append a message to the failure status, e.g. "Failed (<message>)"
                error = self._job["failure"]["error"]
                status += f" ({error})"
            raise gss.SuperstaqUnsuccessfulJobException(self._job_id, status)

    def job_id(self) -> str:
        """Gets the job id of this job.

        This is the unique identifier used for identifying the job by the API.

        Returns:
            This job's id (a string).
        """
        return self._job_id

    def status(self) -> str:
        """Gets the current status of the job.

        If the current job is in a non-terminal state, this will update the job and return the
        current status. A full list of states is given in `cirq_superstaq.Job.ALL_STATES`.

        Raises:
            SuperstaqException: If the API is not able to get the status of the job.

        Returns:
            The job status.
        """
        self._refresh_job()
        return self._job["status"]

    def target(self) -> str:
        """Gets the Superstaq target associated with this job.

        Returns:
            The target to which this job was submitted.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job["target"]

    def num_qubits(self) -> int:
        """Gets the number of qubits required for this job.

        Returns:
            The number of qubits used in this job.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job["num_qubits"]

    def repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job["shots"]

    def counts(self, timeout_seconds: int = 7200, polling_seconds: float = 1.0) -> Dict[str, int]:
        """Polls the Superstaq API for counts results (frequency of each measurement outcome).

        Args:
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.

        Returns:
            A dictionary containing the frequency counts of the measurements.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqException: If unable to get the results from the API.
            TimeoutError: If no results are available in the provided timeout interval.
        """
        time_waited_seconds: float = 0.0
        while self.status() not in self.TERMINAL_STATES:
            # Status does a refresh.
            if time_waited_seconds > timeout_seconds:
                raise TimeoutError(
                    f"Timed out while waiting for results. Final status was {self.status()}"
                )
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds

        self._check_if_unsuccessful()
        return self._job["samples"]

    def __str__(self) -> str:
        return f"Job with job_id={self.job_id()}"

    def __repr__(self) -> str:
        return f"css.Job(client={self._client!r}, job_id={self.job_id()!r})"

    def _value_equality_values_(self) -> Tuple[str, Dict[str, Any]]:
        return self._job_id, self._job
