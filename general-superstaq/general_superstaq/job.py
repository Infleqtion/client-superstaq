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

"""Represents a job created via the Superstaq API."""

from __future__ import annotations

import time
import uuid
from typing import Self

import general_superstaq as gss
from general_superstaq.superstaq_client import _SuperstaqClientV3


class Job:
    """A job created on the Superstaq API.

    Note that this is mutable, when calls to get status or results are made the job updates itself
    If a job is canceled or deleted, only the job id and the status remain valid.
    """

    STATUS_PRIORITY_ORDER = (
        gss.models.CircuitStatus.RECEIVED,
        gss.models.CircuitStatus.AWAITING_COMPILE,
        gss.models.CircuitStatus.AWAITING_SUBMISSION,
        gss.models.CircuitStatus.AWAITING_SIMULATION,
        gss.models.CircuitStatus.COMPILING,
        gss.models.CircuitStatus.SIMULATING,
        gss.models.CircuitStatus.RUNNING,
        gss.models.CircuitStatus.PENDING,
        gss.models.CircuitStatus.FAILED,
        gss.models.CircuitStatus.CANCELLED,
        gss.models.CircuitStatus.UNRECOGNIZED,
        gss.models.CircuitStatus.COMPLETED,
        gss.models.CircuitStatus.DELETED,
    )

    def __init__(
        self,
        client: gss.superstaq_client._SuperstaqClientV3,
        job_id: uuid.UUID | str,
    ) -> None:
        """Constructs a `Job`.

        Users should not call this themselves. If you only know the `job_id`, use `fetch_jobs`
        on `gss.Service`.

        Args:
            client: The client used for calling the API.
            job_id: Unique identifier for the job.

        Raises:
            TypeError: If `JobV3` is used with `v0.2.0` of the Superstaq API.
        """
        if not isinstance(client, _SuperstaqClientV3):
            raise TypeError("JobV3 job can only be used with v0.3.0 of the Superstaq API.")

        self._client = client
        self._overall_status = gss.models.CircuitStatus.RECEIVED
        self._job_data: gss.models.JobData | None = None
        self._job_id = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._job_id == other._job_id

    def __hash__(self) -> int:
        return hash(self._job_id)

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""
        if self._job_data is not None:
            if all(s in gss.models.TERMINAL_CIRCUIT_STATES for s in self._job_data.statuses):
                return
        self._job_data = gss.models.JobData(
            **self._client.fetch_jobs([self._job_id])[str(self._job_id)]
        )
        self._update_status_queue_info()

    def wait_until_complete(
        self,
        index: int | None = None,
        timeout_seconds: float | None = None,
        polling_seconds: float = 5,
    ) -> None:
        """Waits until either the job is done or some error in the job occurs.

        Args:
            index: An optional index of the specific circuit to wait for (otherwise waits for all
                circuits to complete).
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.
        """
        time_waited: float = 0.0
        timeout_seconds = timeout_seconds or self._client.max_retry_seconds
        while (status := self._status(index)) not in gss.models.TERMINAL_CIRCUIT_STATES:
            if time_waited > timeout_seconds:
                raise TimeoutError(
                    f"Timed out while waiting for results. Final status was '{status}'"
                )
            time.sleep(polling_seconds)
            time_waited += polling_seconds

    @property
    def job_data(self) -> gss.models.JobData:
        if self._job_data is None:
            self._refresh_job()
        if self._job_data is None:
            raise AttributeError("Job data has not been fetched yet. Run _refresh_job().")
        return self._job_data

    @property
    def tags(self) -> list[str]:
        """All tags associated with this job."""
        return self.job_data.tags

    @property
    def metadata(self) -> dict[str, object]:
        """Any metadata passed when creating this job."""
        return self.job_data.metadata

    def _update_status_queue_info(self) -> None:
        """Updates the overall status based on status queue info.

        Note:
            When we have multiple jobs, we will take the "positive status" among the jobs. The
            status check sequentially follows the items in `STATUS_PRIORITY_ORDER`. For example,
            if any of the jobs are still running (even if some are done), we report 'Running' as
            the overall status of the entire batch.
        """
        status_occurrence = set(self.job_data.statuses)

        for temp_status in self.STATUS_PRIORITY_ORDER:
            if temp_status in status_occurrence:
                self._overall_status = temp_status
                return

    def _check_if_unsuccessful(self, index: int | None = None) -> None:
        """Helper method to check if the current circuit status has any failure.

        Args:
            index: The index of the specific job status.

        Raises:
            gss.SuperstaqUnsuccessfulJobException: If a failure status is found in the job.
        """
        status = self._status(index)
        if status == gss.models.CircuitStatus.FAILED:
            message = "Failure: "
            circuit_messages = []
            if index is None:
                to_check = list(range(self.job_data.num_circuits))
            else:
                to_check = [index]
            for k in to_check:
                if self.job_data.statuses[k] == gss.models.CircuitStatus.FAILED:
                    error = (
                        self.job_data.status_messages[k]
                        if self.job_data.status_messages[k] is not None
                        else "Unknown"
                    )
                    circuit_messages.append(f"Circuit {k} - {error}")
            message += "[" + ", ".join(circuit_messages) + "]"
            raise gss.SuperstaqUnsuccessfulJobException(str(self.job_id()), message)

    def job_id(self) -> uuid.UUID:
        """Gets the job id of this job.

        This is the unique identifier used for identifying the job by the API.

        Returns:
            This job's id (a string).
        """
        return self._job_id

    def _status(self, index: int | None = None) -> gss.models.CircuitStatus:
        """Gets the current status of the job.

        If the current job is in a non-terminal state, this will update the job and return the
        current status.

        Args:
            index: An optional index of the specific sub-job to get the status of.

        Raises:
            ~gss.SuperstaqServerException: If unable to get the status of the job
               from the API.

        Returns:
            The status of the job indexed by `index` or the overall job status if `index` is `None`.
        """
        self._refresh_job()
        if index is None:
            return self._overall_status

        gss.validation.validate_integer_param(index, min_val=0)
        return self.job_data.statuses[index]

    def cancel(self, **kwargs: object) -> None:
        """Cancel the current job if it is not in a terminal state.

        Args:
            index: An optional index of the specific sub-job to cancel.
            kwargs: Extra options needed to fetch jobs.

        Raises:
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API or cancellations were unsuccessful.
        """
        self._client.cancel_jobs([self._job_id], **kwargs)

    def target(self) -> str:
        """Gets the Superstaq target associated with this job.

        Returns:
            The target to which this job was submitted.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API.
        """
        return self.job_data.target

    def repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.
        """
        return self.job_data.shots[0]

    def to_dict(self) -> dict[str, gss.typing.Job]:
        """Refreshes and returns job information.

        Note:
            The contents of this dictionary are not guaranteed to be consistent over time. Whenever
            possible, users should use the specific `Job` methods to retrieve the desired job
            information instead of relying on particular entries in the output of this method.

        Returns:
            A dictionary containing updated job information.
        """
        return self.job_data.model_dump()

    def __str__(self) -> str:
        return f"Job with job_id={self.job_id()}"

    def __getitem__(self, index: int) -> Job:
        """Customized indexing operations for jobs.

        Args:
            index: The index of the sub-job to return. Each sub-job corresponds to the a single
                circuit.

        Returns:
            A sub-job at the given `index`.
        """
        raise NotImplementedError
