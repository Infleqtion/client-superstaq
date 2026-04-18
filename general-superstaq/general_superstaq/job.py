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

import enum
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

import general_superstaq as gss

if TYPE_CHECKING:
    import jaqalpaq.run.result


class Endian(str, enum.Enum):
    """Endianness to use when mapping quantum objects to matrices, state vectors, and bitstrings."""

    BIG = "big"  # BQSKit, Braket, Cirq, Jaqal, TKET
    LITTLE = "little"  # Qiskit


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

    endianness = Endian.BIG

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
        """
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
            to_check = list(range(self.job_data.num_circuits)) if index is None else [index]
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
            ~gss.SuperstaqServerException: If unable to get the status of the job from the API.

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
            ~gss.SuperstaqServerException: If unable to get the status of the job from the API or
            cancellations were unsuccessful.
        """
        self._client.cancel_jobs([self._job_id], **kwargs)

    def target(self) -> str:
        """Gets the Superstaq target associated with this job.

        Returns:
            The target to which this job was submitted.

        Raises:
            ~gss.SuperstaqServerException: If unable to get the job from the API.
        """
        return self.job_data.target

    def _repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.
        """
        return self.job_data.shots[0]

    def set_counts(
        self,
        results: Sequence[Mapping[str, float]]
        | jaqalpaq.run.result.ExecutionResult
        | jaqalpaq.run.result.SubbatchView,
    ) -> None:
        """Manually input experimental counts for all circuits in this job.

        Also accepts JaqalPaq `ExecutionResult` or `SubbatchView` objects. Both are assumed to have
        as many subcircuits as this job contains compiled circuits. For `ExecutionResult` objects
        containing multiple sub-batches, only the first is used. (requires `jaqalpaq>=1.3.0`)

        Args:
            results: A sequence of experimental counts dictionaries to load, each of which should be
                formatted as required by `Job.set_counts_for_circuit`. Can also be one of the
                JaqalPaq results objects described above. In either case, assumed to contain counts
                for every compiled circuit contained in this job.
        """
        # Support JaqalPaq's `ExecutionResult` and `SubbatchView`:
        results = getattr(results, "by_subbatch", [results])[0]
        results = getattr(results, "by_subcircuit", results)

        for i in range(self.job_data.num_circuits):
            self.set_counts_for_circuit(i, results[i])

    def set_counts_for_circuit(
        self,
        index: int,
        result: Mapping[str, int] | jaqalpaq.run.result.SubcircuitView,
    ) -> None:
        """Manually input experimental counts for one circuits in this job.

        Also accepts JaqalPaq `SubcircuitView`, e.g. from `result.by_subbatch[i].by_subcircuit[j]`
        where `result` is a JaqalPaq `ExecutionResult` object. (requires `jaqalpaq>=1.3.0`)

        Args:
            index: The circuit index for which to update counts.
            result: An experimental counts dictionary to load (or a JaqalPaq `SubcircuitView`).
                Counts dictionaries must be formatted with bitstrings as keys and total counts (or
                probabilities) as values, e.g. `{"000": 100, "111": 200}`.
        """
        # Special handling for JaqalPaq's `SubcircuitView`
        if normalized_counts := getattr(result, "normalized_counts", None):
            num_repeats = getattr(result, "num_repeats", 1)
            result = {
                bs: round(np.dot(prob, num_repeats).sum())
                for bs, prob in normalized_counts.by_str.items()
            }

            # JaqalPaq results are always ordered by qubit (big-endian)
            bitstring_qubit_indices = self._terminal_measurement_qubit_indices(index)
            result = {
                "".join(bs[i] for i in bitstring_qubit_indices): val for bs, val in result.items()
            }

        elif self.endianness == Endian.LITTLE:
            result = {bs[::-1]: val for bs, val in result.items()}

        self.job_data.counts[index] = dict(result)
        self.job_data.counts = self.job_data.counts  # Trigger revalidation

    def _terminal_measurement_qubit_indices(self, index: int) -> list[int]:
        """Determines the ordered physical qubit indices for each measurement in a compiled circuit.

        Assumes all measurements are terminal.

        Args:
            index: The index of the compiled circuit for which to return qubit indices.

        Returns:
            A list of measured qubit indices, ordered as they should appear in (big-endian)
            bitstrings.
        """
        logical_to_physical = self.job_data.final_logical_to_physicals[index]
        if logical_to_physical is None:
            raise ValueError("Circuit must be compiled before counts are set.")

        return [logical_to_physical[i] for i in sorted(logical_to_physical.keys())]

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
            index: The index of the sub-job to return. Each sub-job corresponds to a single circuit.

        Returns:
            A sub-job at the given `index`.
        """
        raise NotImplementedError
