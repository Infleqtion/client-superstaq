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

import time
from typing import Any, Dict, List, Optional

import general_superstaq as gss
import qiskit

import qiskit_superstaq as qss


class SuperstaQJob(qiskit.providers.JobV1):  # pylint: disable=missing-class-docstring

    TERMINAL_STATES = ("Done", "Canceled", "Failed", "Deleted")
    STOPPED_STATES = ("Canceled", "Failed", "Deleted")
    PROCESSING_STATES = ("Queued", "Submitted", "Running")
    ALL_STATES = TERMINAL_STATES + PROCESSING_STATES

    def __init__(self, backend: qss.SuperstaQBackend, job_id: str) -> None:
        """Initialize a job instance.

        Args:
            backend: The `qss.SuperstaQBackend` that the job was created with.
            job_id: The unique job ID from SuperstaQ.
        """
        super().__init__(backend, job_id)
        self._job_info: Dict[str, Any] = {"status": "Submitted"}

    def __eq__(self, other: Any) -> bool:

        if not (isinstance(other, SuperstaQJob)):
            return False

        return self._job_id == other._job_id

    def get_job_id(self) -> str:
        """Returns the virtual job id for the job."""
        return self._job_id

    def get_backend(self) -> str:
        """Returns the job backend used as a string."""
        self._check_if_stopped()
        return self._backend

    def _wait_for_results(
        self, timeout: Optional[float] = None, wait: float = 5
    ) -> List[Dict[str, Dict[str, int]]]:
        """Waits for the results till either the job is done or some in the job occurs.

        Args:
            timeout: Time to wait for results. Defaults to None.
            wait: Time to wait before checking again. Defaults to 5.

        Returns:
            Results from the job.

        Raises:
            qiskit.providers.JobTimeoutError: If the elapsed time is greater than the timeout.
            qiskit.providers.JobError: If an error occurred in the job.
        """

        result_list: List[Dict[str, Dict[str, int]]] = []
        job_ids = self._job_id.split(",")  # separate aggregated job_ids

        for jid in job_ids:
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time

                if timeout and elapsed >= timeout:
                    raise qiskit.providers.JobTimeoutError("Timed out waiting for result")

                result = self._backend._provider._client.get_job(jid)
                if result["status"] == "Done":
                    break
                if result["status"] == "Error":
                    raise qiskit.providers.JobError("API returned error:\n" + str(result))
                time.sleep(wait)
            result_list.append(result)
        return result_list

    def result(self, timeout: Optional[float] = None, wait: float = 5) -> qiskit.result.Result:
        """Get the result data of a circuit.

        Args:
            timeout: Time to wait for results. Defaults to None.
            wait: Time to wait before checking again. Defaults to 5.

        Returns:
            Result details from the job.
        """
        results = self._wait_for_results(timeout, wait)

        # create list of result dictionaries
        results_list = []
        for result in results:
            counts = result["samples"]
            if counts:  # change endianess to match Qiskit
                counts = dict((key[::-1], value) for (key, value) in counts.items())
            results_list.append(
                {
                    "success": True,
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
                "success": True,
                "job_id": self._job_id,
            }
        )

    def _check_if_stopped(self) -> None:
        """Verifies that the job status is not in a stopped state and raises an
        exception if it is.

        Raises:
            SuperstaQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            SuperstaQException: If unable to get the status of the job from the API.
        """
        if self._job_info["status"] in self.STOPPED_STATES:
            raise gss.superstaq_exceptions.SuperstaQUnsuccessfulJobException(
                self.get_job_id(), self._job_info["status"]
            )

    def _refresh_job(self) -> None:
        """Queries the server for the current job status"""

        # when we have multiple jobs, we will take the "worst status" among the jobs
        # For example, if any of the jobs are still queued, we report Queued as the status
        # for the entire batch.

        job_id_list = self._job_id.split(",")  # separate aggregated job ids

        for job_id in job_id_list:

            result = self._backend._provider._client.get_job(job_id)
            temp_status = result["status"]
            if temp_status == "Queued":
                self._job_info["status"] = "Queued"
                break
            elif temp_status in self.STOPPED_STATES:
                self._job_info["status"] = "Canceled"
                break
            elif temp_status == "Running":
                self._job_info["status"] = "Running"
            else:
                self._job_info["status"] = "Done"

    def status(self) -> qiskit.providers.jobstatus.JobStatus:
        """Query for the job status.

        Returns:
            The equivalent `qiskit.providers.jobstatus.JobStatus` type.
        """

        qiskit_status = qiskit.providers.jobstatus.JobStatus.INITIALIZING

        if self._job_info["status"] in self.TERMINAL_STATES:
            if self._job_info["status"] == "Done":
                return qiskit.providers.jobstatus.JobStatus.DONE
            else:
                return qiskit.providers.jobstatus.JobStatus.CANCELLED

        self._refresh_job()
        status = self._job_info["status"]

        assert status in self.ALL_STATES

        status_match = {
            "Queued": qiskit.providers.jobstatus.JobStatus.QUEUED,
            "Running": qiskit.providers.jobstatus.JobStatus.RUNNING,
            "Submitted": qiskit.providers.jobstatus.JobStatus.INITIALIZING,
        }

        if status in self.PROCESSING_STATES:
            qiskit_status = status_match.get(status)
        elif status in self.STOPPED_STATES:
            qiskit_status = qiskit.providers.jobstatus.JobStatus.CANCELLED
        else:
            qiskit_status = qiskit.providers.jobstatus.JobStatus.DONE

        return qiskit_status

    def submit(self) -> None:
        raise NotImplementedError("Submit through SuperstaQBackend, not through SuperstaqJob")
