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

from typing import Any, Dict, List, Optional

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

    def result(self, timeout: Optional[float] = None, wait: float = 5) -> qiskit.result.Result:
        """Retrieves the result data associated with a Superstaq job.

        Args:
            timeout: An optional parameter that fixes when result retrieval times out. Units are
                in seconds.
            wait: An optional parameter that sets the interval to check for Superstaq job results.
                Units are in seconds. Defaults to 5.

        Returns:
            A qiskit result object containing job information.
        """
        timeout = timeout or self._backend._provider._client.max_retry_seconds
        results = self._wait_for_results(timeout, wait)

        # create list of result dictionaries
        results_list = []
        for result in results:
            counts = result["samples"]
            if counts:  # change endianess to match Qiskit
                counts = dict((key[::-1], value) for (key, value) in counts.items())
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
            SuperstaqUnsuccessfulJobException: If the job been cancelled or has
        failed.
            SuperstaqException: If unable to get the status of the job from the API.
        """
        if self._overall_status in ("Cancelled", "Failed"):
            raise gss.superstaq_exceptions.SuperstaqUnsuccessfulJobException(
                self._job_id, self._overall_status
            )

    def _refresh_job(self) -> None:
        """Queries the server for an updated job result."""

        for job_id in self._job_id.split(","):

            if (job_id not in self._job_info) or (
                job_id in self._job_info
                and self._job_info[job_id]["status"] not in self.TERMINAL_STATES
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
            NotImplementedError: If a job is submitted via SuperstaqJob.
        """
        raise NotImplementedError("Submit through SuperstaqBackend, not through SuperstaqJob")
