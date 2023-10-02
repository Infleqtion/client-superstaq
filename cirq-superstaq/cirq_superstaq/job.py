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
import collections
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload

import cirq
import general_superstaq as gss
from cirq._doc import document

import cirq_superstaq as css


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

    NON_TERMINAL_STATES = ("Ready", "Submitted", "Running", "Queued")
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
        """Construct a `Job`.

        Users should not call this themselves. If you only know the `job_id`, use `get_job`
        on `css.Service`.

        Args:
            client: The client used for calling the API.
            job_id: Unique identifier for the job.
        """
        self._client = client
        self._overall_status = "Submitted"
        self._job: Dict[str, Any] = {}
        self._job_id = job_id

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""

        for job_id in self._job_id.split(","):
            if (job_id not in self._job) or (
                self._job[job_id]["status"] not in self.TERMINAL_STATES
            ):
                result = self._client.get_job(job_id)
                self._job[job_id] = result

    def _update_status_queue_info(self) -> None:
        """Updates the overall status based on status queue info.

        Note:
            When we have multiple jobs, we will take the "worst status" among the jobs.
            The worst status check follows the chain: Submitted -> Ready -> Queued ->
            Running -> Failed -> Canceled -> Deleted -> Done. For example, if any of the
            jobs are still running (even if some are done), we report 'Running' as the
            overall status of the entire batch.
        """

        job_id_list = self._job_id.split(",")  # separate aggregated job ids

        status_occurrence = {self._job[job_id]["status"] for job_id in job_id_list}
        status_priority_order = (
            "Submitted",
            "Ready",
            "Queued",
            "Running",
            "Failed",
            "Canceled",
            "Deleted",
            "Done",
        )

        for temp_status in status_priority_order:
            if temp_status in status_occurrence:
                self._overall_status = temp_status
                return

    def _check_if_unsuccessful(self) -> None:
        status = self.status()
        if status in self.UNSUCCESSFUL_STATES:
            for job_id in self._job_id.split(","):
                if "failure" in self._job[job_id] and "error" in self._job[job_id]["failure"]:
                    # if possible append a message to the failure status, e.g. "Failed (<message>)"
                    error = self._job[job_id]["failure"]["error"]
                    status += f" ({error})"
                raise gss.SuperstaqUnsuccessfulJobException(job_id, status)

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
            SuperstaqServerException: If unable to get the status of the job from the API.

        Returns:
            The job status.
        """
        self._refresh_job()
        self._update_status_queue_info()
        return self._overall_status

    def target(self) -> str:
        """Gets the Superstaq target associated with this job.

        Returns:
            The target to which this job was submitted.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqServerException: If unable to get the status of the job from the API.
        """
        first_job_id = self._job_id.split(",")[0]
        if (first_job_id not in self._job) or "target" not in self._job[first_job_id]:
            self._refresh_job()
        return self._job[first_job_id]["target"]

    @overload
    def num_qubits(self, index: int) -> int:
        ...

    @overload
    def num_qubits(
        self, index: None = None
    ) -> Union[int, List[int]]:  # Change return to `List[int]` after deprecation
        ...

    def num_qubits(self, index: Optional[int] = None) -> Union[int, List[int]]:
        """Gets the number of qubits required for each circuit in this job.

        Args:
            index: The index of the circuit to get number of qubits from.

        Returns:
            A list of the numbers of qubits in all circuits or just a single qubit
            number for the given circuit index.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqServerException: If unable to get the status of the job from the API.
        """
        job_ids = self._job_id.split(",")
        if not all(
            job_id in self._job and self._job[job_id].get("num_qubits") for job_id in job_ids
        ):
            self._refresh_job()

        if index is None:
            qubit_list = [self._job[job_id]["num_qubits"] for job_id in job_ids]
            if len(qubit_list) == 1:
                warnings.warn(
                    "In the future, calling `num_qubits()` without an argument will return a list "
                    "of the numbers of qubits in all circuits in this job. Use e.g., "
                    "`num_qubits(0)` to get the number of qubits in the first (or a single) "
                    "circuit.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return qubit_list[0]
            return qubit_list

        gss.validation.validate_integer_param(index, min_val=0)
        num_qubits = self._job[job_ids[index]]["num_qubits"]
        return num_qubits

    def repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqServerException: If unable to get the status of the job from the API.
        """
        first_job_id = self._job_id.split(",")[0]
        if (first_job_id not in self._job) or "shots" not in self._job[first_job_id]:
            self._refresh_job()

        return self._job[first_job_id]["shots"]

    @overload
    def _get_circuits(self, circuit_type: str, index: int) -> cirq.Circuit:
        ...

    @overload
    def _get_circuits(
        self, circuit_type: str, index: None = None
    ) -> Union[
        cirq.Circuit, List[cirq.Circuit]
    ]:  # Change return to `List[cirq.Circuit]` after deprecation
        ...

    def _get_circuits(
        self, circuit_type: str, index: Optional[int] = None
    ) -> Union[cirq.Circuit, List[cirq.Circuit]]:
        """Retrieves the corresponding circuit(s) to `circuit_type`.

        Args:
            circuit_type: The kind of circuit(s) to retrieve. Either "input_circuit" or
                "compiled_circuit".
            index: The index of the circuit to retrieve.

        Returns:
            A single circuit or list of circuits.
        """
        if circuit_type not in ("input_circuit", "compiled_circuit"):
            raise ValueError("The circuit type requested is invalid.")

        job_ids = self._job_id.split(",")
        if not all(
            job_id in self._job and self._job[job_id].get(circuit_type) for job_id in job_ids
        ):
            self._refresh_job()

        if index is None:
            serialized_circuits = [self._job[job_id][circuit_type] for job_id in job_ids]
            return [css.deserialize_circuits(serialized)[0] for serialized in serialized_circuits]

        gss.validation.validate_integer_param(index, min_val=0)
        serialized_circuit = self._job[job_ids[index]][circuit_type]
        return css.deserialize_circuits(serialized_circuit)[0]

    @overload
    def compiled_circuits(self, index: int) -> cirq.Circuit:
        ...

    @overload
    def compiled_circuits(
        self, index: None = None
    ) -> Union[
        cirq.Circuit, List[cirq.Circuit]
    ]:  # Change return to `List[cirq.Circuit]` after deprecation
        ...

    def compiled_circuits(
        self, index: Optional[int] = None
    ) -> Union[cirq.Circuit, List[cirq.Circuit]]:
        """Gets the compiled circuits that were processed for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single compiled circuit or list of compiled circuits.
        """
        return self._get_circuits("compiled_circuit", index=index)

    @overload
    def input_circuits(self, index: int) -> cirq.Circuit:
        ...

    @overload
    def input_circuits(
        self, index: None = None
    ) -> Union[
        cirq.Circuit, List[cirq.Circuit]
    ]:  # Change return to `List[cirq.Circuit]` after deprecation
        ...

    def input_circuits(
        self, index: Optional[int] = None
    ) -> Union[cirq.Circuit, List[cirq.Circuit]]:
        """Gets the original circuits that were submitted for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single input circuit or list of submitted input circuits.
        """
        return self._get_circuits("input_circuit", index=index)

    @overload
    def counts(
        self,
        index: int,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, int]:
        ...

    @overload
    def counts(
        self,
        index: None = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Optional[Sequence[int]] = None,
    ) -> Union[
        Dict[str, int], List[Dict[str, int]]
    ]:  # Change return to just `List[Dict[str, int]]` after deprecation
        ...

    def counts(
        self,
        index: Optional[int] = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Optional[Sequence[int]] = None,
    ) -> Union[Dict[str, int], List[Dict[str, int]]]:
        """Polls the Superstaq API for counts results (frequency of each measurement outcome).

        Args:
            index: The index of the circuit which the counts correspond to.
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.
            qubit_indices: If provided, only include measurements counts of these qubits.

        Returns:
            A dictionary containing the frequency counts of the measurements.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job failed or has been canceled or deleted.
            SuperstaqServerException: If unable to get the results from the API.
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
        job_ids = self._job_id.split(",")

        if index is None:
            counts_list = [self._job[job_id]["samples"] for job_id in self._job_id.split(",")]
            if qubit_indices:
                counts_list = [
                    _get_marginal_counts(counts, qubit_indices) for counts in counts_list
                ]
            if len(counts_list) == 1:
                warnings.warn(
                    "In the future, calling `counts()` without an argument will return a list of "
                    "the counts in all circuits in this job. Use e.g., `counts(0)` to get the "
                    "counts for the first (or a single) circuit.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return counts_list[0]
            return counts_list

        gss.validation.validate_integer_param(index, min_val=0)
        single_counts = self._job[job_ids[index]]["samples"]
        if qubit_indices:
            return _get_marginal_counts(single_counts, qubit_indices)
        return single_counts

    def to_dict(self) -> Dict[str, gss.typing.Job]:
        """Refreshes and returns job information.

        Note:
            The contents of this dictionary are not guaranteed to be consistent over time. Whenever
            possible, users should use the specific `Job` methods to retrieve the desired job
            information instead of relying on particular entries in the output of this method.

        Returns:
            A dictionary containing updated job information.
        """
        self._refresh_job()
        return self._job

    def __str__(self) -> str:
        return f"Job with job_id={self.job_id()}"

    def __repr__(self) -> str:
        return f"css.Job(client={self._client!r}, job_id={self.job_id()!r})"

    def _value_equality_values_(self) -> Tuple[str, Dict[str, Any]]:
        return self._job_id, self._job


def _get_marginal_counts(counts: Dict[str, int], indices: Sequence[int]) -> Dict[str, int]:
    """Compute a marginal distribution, accumulating total counts on specific bits (by index).

    Args:
        counts: The dictionary containing all the counts.
        indices: The indices of the bits on which to accumulate counts.

    Returns:
        A dictionary of counts on the target indices.
    """
    target_counts: Dict[str, int] = collections.defaultdict(int)
    for bitstring, count in counts.items():
        target_key = "".join([bitstring[index] for index in indices])
        target_counts[target_key] += count
    return dict(target_counts)
