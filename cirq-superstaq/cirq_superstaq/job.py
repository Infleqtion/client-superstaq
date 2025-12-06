# Copyright 2025 Infleqtion
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

from __future__ import annotations

import collections
import time
import uuid
from collections.abc import Sequence
from typing import Any, overload

import cirq
import general_superstaq as gss
from cirq._doc import document
from general_superstaq.superstaq_client import _SuperstaqClientV3

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
        """Constructs a `Job`.

        Users should not call this themselves. If you only know the `job_id`, use `fetch_jobs`
        on `css.Service`.

        Args:
            client: The client used for calling the API.
            job_id: Unique identifier for the job.
        """
        self._client = client
        self._overall_status = "Submitted"
        self._job: dict[str, Any] = {}
        self._job_id = job_id

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""
        jobs_to_fetch: list[str] = []

        for job_id in self._job_id.split(","):
            if job_id not in self._job or self._job[job_id]["status"] not in self.TERMINAL_STATES:
                jobs_to_fetch.append(job_id)

        if jobs_to_fetch:
            result = self._client.fetch_jobs(jobs_to_fetch)
            self._job.update(result)

        self._update_status_queue_info()

    def _update_status_queue_info(self) -> None:
        """Updates the overall status based on status queue info.

        Note:
            When we have multiple jobs, we will take the "positive status" among the jobs. The
            status check follows the chain: Submitted -> Ready -> Running -> Queued -> Failed ->
            Canceled -> Deleted -> Done. For example, if any of the jobs are still running (even
            if some are done), we report 'Running' as the overall status of the entire batch.
        """
        job_id_list = self._job_id.split(",")  # Separate aggregated job ids

        status_occurrence = {self._job[job_id]["status"] for job_id in job_id_list}
        status_priority_order = (
            "Submitted",
            "Ready",
            "Running",
            "Queued",
            "Failed",
            "Canceled",
            "Deleted",
            "Done",
        )

        for temp_status in status_priority_order:
            if temp_status in status_occurrence:
                self._overall_status = temp_status
                return

    def _check_if_unsuccessful(self, index: int | None = None) -> None:
        """Helper method to check if the current job status has any failure.

        Args:
            index: The index of the specific job status.

        Raises:
            gss.SuperstaqUnsuccessfulJobException: If a failure status is found in the job.
        """
        status = self.status(index)
        if status in self.UNSUCCESSFUL_STATES:
            job_ids = self._job_id.split(",")
            ids_to_check = [job_ids[index]] if index else job_ids
            for job_id in ids_to_check:
                if "failure" in self._job[job_id] and "error" in self._job[job_id]["failure"]:
                    # If possible append a message to the failure status, e.g. "Failed (<message>)"
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

    def status(self, index: int | None = None) -> str:
        """Gets the current status of the job.

        If the current job is in a non-terminal state, this will update the job and return the
        current status. A full list of states is given in `cirq_superstaq.Job.ALL_STATES`.

        Args:
            index: An optional index of the specific sub-job to get the status of.

        Raises:
            ~gss.SuperstaqServerException: If unable to get the status of the job
               from the API.

        Returns:
            The status of the job indexed by `index` or the overall job status if `index` is `None`.
        """
        if index is None:
            self._refresh_job()
            return self._overall_status

        gss.validation.validate_integer_param(index, min_val=0)

        job_ids = self._job_id.split(",")
        requested_job_id = job_ids[index]
        if (
            requested_job_id not in self._job
            or self._job[requested_job_id]["status"] not in self.TERMINAL_STATES
        ):
            self._job.update(self._client.fetch_jobs([requested_job_id]))
        requested_job_status = self._job[requested_job_id]["status"]
        return requested_job_status

    def cancel(self, index: int | None = None, **kwargs: object) -> None:
        """Cancel the current job if it is not in a terminal state.

        Args:
            index: An optional index of the specific sub-job to cancel.
            kwargs: Extra options needed to fetch jobs.

        Raises:
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API or cancellations were unsuccessful.
        """
        job_ids = self._job_id.split(",")
        ids_to_cancel = [job_ids[index]] if index else job_ids
        self._client.cancel_jobs(ids_to_cancel, **kwargs)

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
        first_job_id = self._job_id.split(",")[0]
        if (first_job_id not in self._job) or "target" not in self._job[first_job_id]:
            self._refresh_job()
        return self._job[first_job_id]["target"]

    @overload
    def num_qubits(self, index: int) -> int: ...

    @overload
    def num_qubits(self, index: None = None) -> list[int]: ...

    def num_qubits(self, index: int | None = None) -> int | list[int]:
        """Gets the number of qubits required for each circuit in this job.

        Args:
            index: The index of the circuit to get number of qubits from.

        Returns:
            A list of the numbers of qubits in all circuits or just a single qubit
            number for the given circuit index.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API.
        """
        job_ids = self._job_id.split(",")
        if not all(
            job_id in self._job and self._job[job_id].get("num_qubits") for job_id in job_ids
        ):
            self._refresh_job()

        if index is None:
            return [self._job[job_id]["num_qubits"] for job_id in job_ids]

        gss.validation.validate_integer_param(index, min_val=0)
        num_qubits = self._job[job_ids[index]]["num_qubits"]
        return num_qubits

    def repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API.
        """
        first_job_id = self._job_id.split(",")[0]
        if (first_job_id not in self._job) or "shots" not in self._job[first_job_id]:
            self._refresh_job()

        return self._job[first_job_id]["shots"]

    @overload
    def _get_circuits(self, circuit_type: str, index: int) -> cirq.Circuit: ...

    @overload
    def _get_circuits(self, circuit_type: str, index: None = None) -> list[cirq.Circuit]: ...

    def _get_circuits(
        self, circuit_type: str, index: int | None = None
    ) -> cirq.Circuit | list[cirq.Circuit]:
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
    def compiled_circuits(self, index: int) -> cirq.Circuit: ...

    @overload
    def compiled_circuits(self, index: None = None) -> list[cirq.Circuit]: ...

    def compiled_circuits(self, index: int | None = None) -> cirq.Circuit | list[cirq.Circuit]:
        """Gets the compiled circuits that were processed for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single compiled circuit or list of compiled circuits.
        """
        return self._get_circuits("compiled_circuit", index=index)

    @overload
    def input_circuits(self, index: int) -> cirq.Circuit: ...

    @overload
    def input_circuits(self, index: None = None) -> list[cirq.Circuit]: ...

    def input_circuits(self, index: int | None = None) -> cirq.Circuit | list[cirq.Circuit]:
        """Gets the original circuits that were submitted for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single input circuit or list of submitted input circuits.
        """
        return self._get_circuits("input_circuit", index=index)

    def pulse_gate_circuits(self, index: int | None = None) -> Any:
        """Gets the pulse gate circuit(s) returned by this job.

        Args:
            index: An optional index of the pulse gate circuit to retrieve.

        Returns:
            A `qiskit.QuantumCircuit` pulse gate circuit or list of `qiskit.QuantumCircuit` pulse
            gate circuits.

        Raises:
            ValueError: If the job was not run on an IBM pulse device.
            ValueError: If the job's target does not use pulse gate circuits.
        """
        job_ids = self._job_id.split(",")

        if not all(
            job_id in self._job and self._job[job_id].get("pulse_gate_circuits") is not None
            for job_id in job_ids
        ):
            self._refresh_job()

        if index is None:
            if all(self._job[job_id].get("pulse_gate_circuits") for job_id in job_ids):
                serialized_circuits = [
                    self._job[job_id]["pulse_gate_circuits"] for job_id in job_ids
                ]
                deserialized_circuits = []
                for serialized_circuit in serialized_circuits:
                    deserialized_circuit = css.serialization.deserialize_qiskit_circuits(
                        serialized_circuit, False
                    )
                    if deserialized_circuit is None:
                        raise ValueError("Some circuits could not be deserialized.")
                    deserialized_circuits.append(deserialized_circuit[0])
                return deserialized_circuits
        else:
            gss.validation.validate_integer_param(index, min_val=0)
            serialized_circuit = self._job[job_ids[index]]["pulse_gate_circuits"]
            return css.serialization.deserialize_qiskit_circuits(serialized_circuit, False)

        error = f"Target '{self.target()}' does not use pulse gate circuits."
        raise ValueError(error)

    @overload
    def counts(
        self,
        index: int,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> dict[str, int]: ...

    @overload
    def counts(
        self,
        index: None = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> list[dict[str, int]]: ...

    def counts(
        self,
        index: int | None = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> dict[str, int] | list[dict[str, int]]:
        """Polls the Superstaq API for counts results (frequency of each measurement outcome).

        Args:
            index: The index of the circuit which the counts correspond to.
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.
            qubit_indices: If provided, only include measurements counts of these qubits.

        Returns:
            A dictionary containing the frequency counts of the measurements the job indexed by
            `index` or a list of such dictionaries for each respective sub-job.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the results from the API.
            TimeoutError: If no results are available in the provided timeout interval.
        """
        time_waited_seconds: float = 0.0
        while (status := self.status(index)) not in self.TERMINAL_STATES:
            # Status does a refresh.
            if time_waited_seconds > timeout_seconds:
                raise TimeoutError(
                    f"Timed out while waiting for results. Final status was '{status}'"
                )
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds

        self._check_if_unsuccessful(index)
        job_ids = self._job_id.split(",")

        if index is None:
            counts_list = [self._job[job_id]["samples"] for job_id in self._job_id.split(",")]
            if qubit_indices:
                counts_list = [
                    _get_marginal_counts(counts, qubit_indices) for counts in counts_list
                ]
            return counts_list

        gss.validation.validate_integer_param(index, min_val=0)
        single_counts = self._job[job_ids[index]]["samples"]
        if qubit_indices:
            return _get_marginal_counts(single_counts, qubit_indices)
        return single_counts

    def to_dict(self) -> dict[str, gss.typing.Job]:
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

    def _value_equality_values_(self) -> tuple[str, dict[str, Any] | None]:
        return self._job_id, self._job

    def __getitem__(self, index: int) -> css.Job:
        """Customized indexing operations for `css.Job`.

        Args:
            index: The index of the sub-job to return. Each sub-job corresponds to the a single
                circuit.

        Returns:
            A sub-job at the given `index`.
        """
        job_id = self._job_id.split(",")[index]
        sub_job = css.Job(self._client, job_id)
        if job_id in self._job:
            sub_job._job[job_id] = self._job[job_id]
        return sub_job


@cirq.value_equality(unhashable=True)
class JobV3:
    """***NEW API STANDARD*** A job created on the Superstaq API.

    Note that this is mutable, when calls to get status or results are made the job updates itself
    to the results returned from the API.

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
        on `css.Service`.

        Args:
            client: The client used for calling the API.
            job_id: Unique identifier for the job.

        Raises:
            TypeError: If `JobV3` is used with `v0.2.0` of the Superstaq API.
        """
        self._client = client
        if not isinstance(self._client, _SuperstaqClientV3):
            raise TypeError("JobV3 job can only be used with v0.3.0 of the Superstaq API.")
        self._overall_status = gss.models.CircuitStatus.RECEIVED
        self._job_data: gss.models.JobData | None = None
        self._job_id = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""
        if self._job_data is not None:
            if all(s in gss.models.TERMINAL_CIRCUIT_STATES for s in self._job_data.statuses):
                return
        self._job_data = gss.models.JobData(
            **self._client.fetch_jobs([self._job_id])[str(self._job_id)]
        )
        self._update_status_queue_info()

    @property
    def job_data(self) -> gss.models.JobData:
        if self._job_data is None:
            self._refresh_job()
        if self._job_data is None:
            raise AttributeError("Job data has not been fetched yet. Run _refresh_job().")
        else:
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
        status = self.status(index)
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

    def status(self, index: int | None = None) -> gss.models.CircuitStatus:
        """Gets the current status of the job.

        If the current job is in a non-terminal state, this will update the job and return the
        current status. A full list of states is given in `cirq_superstaq.Job.ALL_STATES`.

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

    @overload
    def num_qubits(self, index: int) -> int: ...

    @overload
    def num_qubits(self, index: None = None) -> list[int]: ...

    def num_qubits(self, index: int | None = None) -> int | list[int]:
        """Gets the number of qubits required for each circuit in this job.

        Args:
            index: The index of the circuit to get number of qubits from.

        Returns:
            A list of the numbers of qubits in all circuits or just a single qubit
            number for the given circuit index.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the status of the job
                from the API.
        """
        num_qubits = []
        if index is None:
            to_check = list(range(self.job_data.num_circuits))
        else:
            to_check = [index]
        for k in to_check:
            circuit = self.input_circuits(index=k)
            num_qubits.append(cirq.num_qubits(circuit))
        if index is None:
            return num_qubits
        else:
            return num_qubits[0]

    def repetitions(self) -> int:
        """Gets the number of repetitions requested for this job.

        Returns:
            The number of repetitions for this job.
        """
        return self.job_data.shots[0]

    @overload
    def compiled_circuits(self, index: int) -> cirq.Circuit: ...

    @overload
    def compiled_circuits(self, index: None = None) -> list[cirq.Circuit]: ...

    def compiled_circuits(self, index: int | None = None) -> cirq.Circuit | list[cirq.Circuit]:
        """Gets the compiled circuits that were processed for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Raises:
            gss.SuperstaqException: If the circuit at the provided index has no compiled circuit or,
            when index is None, if any of the circuits in the job are missing the compiled circuit.

        Returns:
            A single compiled circuit or list of compiled circuits.
        """
        if index is None:
            if all(c is None for c in self.job_data.compiled_circuits):
                raise gss.SuperstaqException(f"The job {self._job_id} has no compiled circuits.")

            if any(c is None for c in self.job_data.compiled_circuits):
                raise gss.SuperstaqException(
                    "Some compiled circuits are missing. "
                    "This is likely because there was an error on the server. "
                    "Please check the individual circuit statuses and any status messages."
                )
        elif self.job_data.compiled_circuits[index] is None:
            raise gss.SuperstaqException(
                f"Circuit {index} of job {self._job_id} does not have a compiled circuit."
            )

        circuits = [
            css.deserialize_circuits(self.job_data.compiled_circuits[k])[  # type: ignore[arg-type]
                0
            ]
            for k in range(self.job_data.num_circuits)
        ]

        if index is None:
            return circuits
        return circuits[index]

    @overload
    def input_circuits(self, index: int) -> cirq.Circuit: ...

    @overload
    def input_circuits(self, index: None = None) -> list[cirq.Circuit]: ...

    def input_circuits(self, index: int | None = None) -> cirq.Circuit | list[cirq.Circuit]:
        """Gets the original circuits that were submitted for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single input circuit or list of submitted input circuits.
        """
        circuits = [
            css.deserialize_circuits(self.job_data.input_circuits[k])[0]
            for k in range(self.job_data.num_circuits)
        ]

        if index is None:
            return circuits
        return circuits[index]

    @overload
    def counts(
        self,
        index: int,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> dict[str, int]: ...

    @overload
    def counts(
        self,
        index: None = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> list[dict[str, int]]: ...

    def counts(
        self,
        index: int | None = None,
        timeout_seconds: int = 7200,
        polling_seconds: float = 1.0,
        qubit_indices: Sequence[int] | None = None,
    ) -> dict[str, int] | list[dict[str, int]]:
        """Polls the Superstaq API for counts results (frequency of each measurement outcome).

        Args:
            index: The index of the circuit which the counts correspond to.
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.
            qubit_indices: If provided, only include measurements counts of these qubits.

        Returns:
            A dictionary containing the frequency counts of the measurements the job indexed by
            `index` or a list of such dictionaries for each respective sub-job.

        Raises:
            ~gss.SuperstaqUnsuccessfulJobException: If the job failed or has been
                canceled or deleted.
            ~gss.SuperstaqServerException: If unable to get the results from the API.
            TimeoutError: If no results are available in the provided timeout interval.
            gss.SuperstaqException: If the job counts are missing.
        """
        time_waited_seconds: float = 0.0
        # If not in a terminal state then poll
        while (status := self.status(index)) not in gss.models.TERMINAL_CIRCUIT_STATES:
            # Status does a refresh.
            if time_waited_seconds > timeout_seconds:
                raise TimeoutError(
                    f"Timed out while waiting for results. Final status was '{status}'"
                )
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds

        # Check to see if unsuccessful
        self._check_if_unsuccessful(index)

        if index is None:
            if any(c is None for c in self.job_data.counts):
                raise gss.SuperstaqException(
                    f"Job {self._job_id} does not have counts for all circuits."
                )
            counts_list: list[dict[str, int]] = self.job_data.counts  # type: ignore[assignment]
            # Type checking does not recognise that the above error catches the case when any of
            # the counts are None.
            if qubit_indices:
                counts_list = [
                    _get_marginal_counts(counts, qubit_indices) for counts in counts_list
                ]
            return counts_list

        gss.validation.validate_integer_param(index, min_val=0)
        single_counts = self.job_data.counts[index]
        if single_counts is None:
            raise gss.SuperstaqException(
                f"Circuit {index} of job {self._job_id} does not have any counts."
            )
        if qubit_indices:
            return _get_marginal_counts(single_counts, qubit_indices)
        return single_counts

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

    def __repr__(self) -> str:
        return f"css.JobV3(client={self._client!r}, job_id={self.job_id()!r})"

    def _value_equality_values_(self) -> tuple[uuid.UUID, dict[str, Any] | None]:
        if self._job_data is None:
            return self._job_id, None
        return self._job_id, self.job_data.model_dump()

    def __getitem__(self, index: int) -> css.Job:
        """Customized indexing operations for `css.Job`.

        Args:
            index: The index of the sub-job to return. Each sub-job corresponds to the a single
                circuit.

        Returns:
            A sub-job at the given `index`.
        """
        raise NotImplementedError


def _get_marginal_counts(counts: dict[str, int], indices: Sequence[int]) -> dict[str, int]:
    """Compute a marginal distribution, accumulating total counts on specific bits (by index).

    Args:
        counts: The dictionary containing all the counts.
        indices: The indices of the bits on which to accumulate counts.

    Returns:
        A dictionary of counts on the target indices.
    """
    target_counts: dict[str, int] = collections.defaultdict(int)
    for bitstring, count in counts.items():
        target_key = "".join([bitstring[index] for index in indices])
        target_counts[target_key] += count
    return dict(target_counts)
