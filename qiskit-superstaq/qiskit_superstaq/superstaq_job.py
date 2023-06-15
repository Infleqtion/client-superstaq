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

from collections import Counter
from typing import Any, Dict, List, Optional

import general_superstaq as gss
import qiskit

import qiskit_superstaq as qss


class SuperstaQJob(qiskit.providers.JobV1):  # pylint: disable=missing-class-docstring

    TERMINAL_STATES = ("Done", "Canceled", "Error")
    PROCESSING_STATES = ("Queued", "Submitted", "Running")
    ALL_STATES = TERMINAL_STATES + PROCESSING_STATES

    def __init__(self, backend: qss.SuperstaQBackend, job_id: str) -> None:
        """Initialize a job instance.

        Args:
            backend: The `qss.SuperstaQBackend` that the job was created with.
            job_id: The unique job ID from Superstaq.
        """
        super().__init__(backend, job_id)
        self._overall_status = "Submitted"
        self._job_info: Dict[str, Any] = {}

    def __eq__(self, other: object) -> bool:

        if not (isinstance(other, SuperstaQJob)):
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
        self._check_if_stopped()

        return [self._job_info[job_id] for job_id in self._job_id.split(",")]

    def result(self, timeout: Optional[float] = None, wait: float = 5) -> qiskit.result.Result:
        """Get the result data of a circuit.

        Args:
            timeout: Time to wait for results. Defaults to None.
            wait: Time to wait before checking again. Defaults to 5.

        Returns:
            Result details from the job.
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
                "success": result["status"] == "Done",
                "job_id": self._job_id,
            }
        )

    def _check_if_stopped(self) -> None:
        """Verifies that the job status is not in a canceled state and raises an
        exception if it is.

        Raises:
            SuperstaQUnsuccessfulJob: If the job been canceled or an error has occured.
            SuperstaQException: If unable to get the status of the job from the API.
        """
        if self._overall_status in ("Canceled", "Error"):
            raise gss.superstaq_exceptions.SuperstaQUnsuccessfulJobException(
                self._job_id, self._overall_status
            )

    def _refresh_job(self) -> None:
        """Queries the server for the current overall job status.

        Note:
            When we have multiple jobs, we will take the "worst status" among the jobs.
            The worst status check follows the chain: Submitted -> Queued -> Running -> Error
            -> Cancelled -> Done. For example, if any of the jobs are still queued (even if
            some are done), we report 'Queued' as the overall status of the entire batch.
        """

        job_id_list = self._job_id.split(",")  # separate aggregated job ids

        for job_id in job_id_list:

            if (job_id not in self._job_info) or (
                job_id in self._job_info
                and self._job_info[job_id]["status"] not in self.TERMINAL_STATES
            ):
                result = self._backend._provider._client.get_job(job_id)
                self._job_info[job_id] = result

        status_occurance = Counter(self._job_info[job_id]["status"] for job_id in job_id_list)
        status_priority_order = ["Submitted", "Queued", "Running", "Error", "Canceled"]

        for temp_status in status_priority_order:
            if status_occurance[temp_status] > 0:
                self._overall_status = temp_status
                break

        if status_occurance["Done"] == len(job_id_list):
            self._overall_status = "Done"

    def status(self) -> qiskit.providers.jobstatus.JobStatus:
        """Query for the equivalent qiskit job status.

        Returns:
            The equivalent `qiskit.providers.jobstatus.JobStatus` type.
        """

        status_match = {
            "Queued": qiskit.providers.jobstatus.JobStatus.QUEUED,
            "Running": qiskit.providers.jobstatus.JobStatus.RUNNING,
            "Submitted": qiskit.providers.jobstatus.JobStatus.INITIALIZING,
            "Canceled": qiskit.providers.jobstatus.JobStatus.CANCELLED,
            "Error": qiskit.providers.jobstatus.JobStatus.ERROR,
            "Done": qiskit.providers.jobstatus.JobStatus.DONE,
        }

        if self._overall_status in self.TERMINAL_STATES:
            return status_match.get(self._overall_status)

        self._refresh_job()
        status = self._overall_status

        assert status in ("Queued", "Running", "Submitted", "Canceled", "Error", "Done")

        return status_match.get(status)

    def submit(self) -> None:
        raise NotImplementedError("Submit through SuperstaQBackend, not through SuperstaQJob")
