# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Any, Dict, List, Optional, Sequence, Union, overload

import general_superstaq as gss
import qiskit

import qiskit_superstaq as qss


class SuperstaqJob(qiskit.providers.JobV1):
    """This class represents a Superstaq job instance."""

    TERMINAL_STATES = ("Done", "Cancelled", "Failed")
    PROCESSING_STATES = ("Queued", "Submitted", "Running")
    ALL_STATES = TERMINAL_STATES + PROCESSING_STATES

    def __init__(self, backend: qss.SuperstaqBackend, job_id: str) -> None:
        """Initialize a job instance.

        Args:
            backend: The `qss.SuperstaqBackend` that the job was created with.
            job_id: The unique job ID string from Superstaq.
        """
        super().__init__(backend, job_id)
        self._overall_status = "Submitted"
        self._job_info: Dict[str, Any] = {}

    def __eq__(self, other: object) -> bool:
        if not (isinstance(other, SuperstaqJob)):
            return False

        return self._job_id == other._job_id

    def _wait_for_results(self, timeout: float, wait: float = 5) -> List[Dict[str, Dict[str, int]]]:
        """Waits for the results till either the job is done or some error in the job occurs.

        Args:
            timeout: Time to wait for results. Defaults to None.
            wait: Time to wait before checking again. Defaults to 5.

        Returns:
            Results from the job.
        """

        self.wait_for_final_state(timeout, wait)  # should call self.status()
        return [self._job_info[job_id] for job_id in self._job_id.split(",")]

    def result(
        self,
        index: Optional[int] = None,
        timeout: Optional[float] = None,
        wait: float = 5,
        qubit_indices: Optional[Sequence[int]] = None,
    ) -> qiskit.result.Result:
        """Retrieves the result data associated with a Superstaq job.

        Args:
            index: An optional index to retrieve a specific result from a result list.
            timeout: An optional parameter that fixes when result retrieval times out. Units are
                in seconds.
            wait: An optional parameter that sets the interval to check for Superstaq job results.
                Units are in seconds. Defaults to 5.
            qubit_indices: The qubit indices to return the results of individually.

        Returns:
            A qiskit result object containing job information.
        """
        if index is not None:
            gss.validation.validate_integer_param(index, min_val=0)
        timeout = timeout or self._backend._provider._client.max_retry_seconds
        job_results = self._wait_for_results(timeout, wait)
        results = job_results if index is None else [job_results[index]]

        results_list = []  # create list of result dictionaries
        for result in results:
            counts = result["samples"]
            if counts:  # match endianness to Qiskit
                counts = dict((key[::-1], value) for (key, value) in counts.items())
                if qubit_indices:
                    counts = qiskit.result.marginal_counts(counts, indices=qubit_indices)
            results_list.append(
                {
                    "success": result["status"] == "Done",
                    "status": result["status"],
                    "shots": result["shots"],
                    "data": {"counts": counts},
                }
            )
        return qiskit.result.Result.from_dict(
            {
                "results": results_list,
                "qobj_id": -1,
                "backend_name": self._backend._configuration.backend_name,
                "backend_version": self._backend._configuration.backend_version,
                "success": self._overall_status == "Done",
                "status": self._overall_status,
                "job_id": self._job_id,
            }
        )

    def _check_if_stopped(self) -> None:
        """Verifies that the job status is not in a cancelled or failed state and
        raises an exception if it is.

        Raises:
            SuperstaqUnsuccessfulJobException: If the job has been cancelled or has
        failed.
            SuperstaqServerException: If unable to get the status of the job from the API.
        """
        if self._overall_status in ("Cancelled", "Failed"):
            raise gss.superstaq_exceptions.SuperstaqUnsuccessfulJobException(
                self._job_id, self._overall_status
            )

    def _refresh_job(self) -> None:
        """Queries the server for an updated job result."""

        for job_id in self._job_id.split(","):
            if (job_id not in self._job_info) or (
                self._job_info[job_id]["status"] not in self.TERMINAL_STATES
            ):
                result = self._backend._provider._client.get_job(job_id)
                self._job_info[job_id] = result

        self._update_status_queue_info()

    def _update_status_queue_info(self) -> None:
        """Updates the overall status based on status queue info.

        Note:
            When we have multiple jobs, we will take the "worst status" among the jobs.
            The worst status check follows the chain: Submitted -> Queued -> Running -> Failed
            -> Cancelled -> Done. For example, if any of the jobs are still queued (even if
            some are done), we report 'Queued' as the overall status of the entire batch.
        """

        job_id_list = self._job_id.split(",")  # separate aggregated job ids

        status_occurrence = {self._job_info[job_id]["status"] for job_id in job_id_list}
        status_priority_order = ("Submitted", "Queued", "Running", "Failed", "Cancelled", "Done")

        for temp_status in status_priority_order:
            if temp_status in status_occurrence:
                self._overall_status = temp_status
                return

    @overload
    def _get_circuits(self, circuit_type: str, index: int) -> qiskit.QuantumCircuit:
        ...

    @overload
    def _get_circuits(self, circuit_type: str, index: None = None) -> List[qiskit.QuantumCircuit]:
        ...

    def _get_circuits(
        self, circuit_type: str, index: Optional[int] = None
    ) -> Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]:
        """Retrieves the corresponding circuit(s) to `circuit_type`.

        Args:
            circuit_type: The kind of circuit(s) to retrieve. Either "input_circuit" or
                "compiled_circuit".
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single circuit or list of circuits.
        """
        if circuit_type not in ("input_circuit", "compiled_circuit"):
            raise ValueError("The circuit type requested is invalid.")

        job_ids = self._job_id.split(",")
        if not all(
            job_id in self._job_info and self._job_info[job_id].get(circuit_type)
            for job_id in job_ids
        ):
            self._refresh_job()

        if index is None:
            serialized_circuits = [self._job_info[job_id][circuit_type] for job_id in job_ids]
            return [qss.deserialize_circuits(serialized)[0] for serialized in serialized_circuits]

        gss.validation.validate_integer_param(index, min_val=0)
        serialized_circuit = self._job_info[job_ids[index]][circuit_type]
        return qss.deserialize_circuits(serialized_circuit)[0]

    @overload
    def compiled_circuits(self, index: int) -> qiskit.QuantumCircuit:
        ...

    @overload
    def compiled_circuits(self, index: None = None) -> List[qiskit.QuantumCircuit]:
        ...

    def compiled_circuits(
        self, index: Optional[int] = None
    ) -> Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]:
        """Gets the compiled circuits that were processed for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            A single compiled circuit or list of compiled circuits.
        """
        compiled_circuits = self._get_circuits("compiled_circuit", index)
        input_circuits = self._get_circuits("input_circuit", index)
        for compiled_qc, in_qc in zip(compiled_circuits, input_circuits):
            compiled_qc.metadata = in_qc.metadata

        return compiled_circuits

    @overload
    def input_circuits(self, index: int) -> qiskit.QuantumCircuit:
        ...

    @overload
    def input_circuits(self, index: None = None) -> List[qiskit.QuantumCircuit]:
        ...

    def input_circuits(
        self, index: Optional[int] = None
    ) -> Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]:
        """Gets the original circuits that were submitted for this job.

        Args:
            index: An optional index of the specific circuit to retrieve.

        Returns:
            The input circuit or list of submitted input circuits.
        """
        return self._get_circuits("input_circuit", index)

    def status(self) -> qiskit.providers.jobstatus.JobStatus:
        """Query for the equivalent qiskit job status.

        Returns:
            The equivalent `qiskit.providers.jobstatus.JobStatus` type.
        """

        status_match = {
            "Queued": qiskit.providers.jobstatus.JobStatus.QUEUED,
            "Running": qiskit.providers.jobstatus.JobStatus.RUNNING,
            "Submitted": qiskit.providers.jobstatus.JobStatus.INITIALIZING,
            "Cancelled": qiskit.providers.jobstatus.JobStatus.CANCELLED,
            "Failed": qiskit.providers.jobstatus.JobStatus.ERROR,
            "Done": qiskit.providers.jobstatus.JobStatus.DONE,
        }

        if self._overall_status in self.TERMINAL_STATES:
            return status_match.get(self._overall_status)

        self._refresh_job()
        status = self._overall_status

        return status_match.get(status)

    def submit(self) -> None:
        """Unsupported submission call.

        Raises:
            NotImplementedError: If a job is submitted via `SuperstaqJob`.
        """
        raise NotImplementedError("Submit through SuperstaqBackend, not through SuperstaqJob")

    def to_dict(self) -> Dict[str, gss.typing.Job]:
        """Refreshes and returns job information.

        Note:
            The contents of this dictionary are not guaranteed to be consistent over time. Whenever
            possible, users should use the specific `SuperstaqJob` methods to retrieve the desired
            job information instead of relying on particular entries in the output of this method.

        Returns:
            A dictionary containing updated job information.
        """
        if self._overall_status not in self.TERMINAL_STATES:
            self._refresh_job()
        return self._job_info
