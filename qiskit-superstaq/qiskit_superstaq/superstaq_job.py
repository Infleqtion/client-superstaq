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

import qiskit

import qiskit_superstaq as qss


class SuperstaQJob(qiskit.providers.JobV1):  # pylint: disable=missing-class-docstring

    TERMINAL_STATES = ("Cancelled", "Done", "Error")

    def __init__(self, backend: qss.SuperstaQBackend, job_id: str) -> None:
        """Initialize a job instance.

        Args:
            backend: The `qss.SuperstaQBackend` that the job was created with.
            job_id: The unique job ID from SuperstaQ.
        """
        super().__init__(backend, job_id)

    def __eq__(self, other: Any) -> bool:

        if not (isinstance(other, SuperstaQJob)):
            return False

        return self._job_id == other._job_id

    def get_job_id(self) -> str:
        """Returns the job id for the job."""
        return self._job_id

    def get_target(self) -> str:
        """Returns the target where the job is to be run, or was run.

        Returns:
            'qpu' or 'simulator' depending on where the job was run or is running.
        """
        return self._job["target"]

    def _wait_for_results(
        self, timeout: Optional[float] = None, wait: float = 5
    ) -> List[Dict[str, Dict[str, int]]]:
        """Waits for the results till either the job is done or some in the job occurs.

        Args:
            timeout: Time to wait for results. Defaults to None.
            wait: Time to wait before checking . Defaults to 5.

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
            wait: Time to wait before checking . Defaults to 5.

        Returns:
            Result details from the job
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

    def status(self) -> qiskit.providers.jobstatus.JobStatus:
        """Query for the job status."""

        job_id_list = self._job_id.split(",")  # separate aggregated job ids

        status = "Done"

        # when we have multiple jobs, we will take the "worst status" among the jobs
        # For example, if any of the jobs are still queued, we report Queued as the status
        # for the entire batch.

        if not all(
            self._backend._provider._client.get_job(job_id)["status"] in self.TERMINAL_STATES
            for job_id in job_id_list
        ):

            for job_id in job_id_list:
                result = self._backend._provider._client.get_job(job_id)
                temp_status = result["status"]

                if temp_status == "Queued":
                    status = "Queued"
                    break
                elif temp_status == "Running":
                    status = "Running"

            assert status in ["Queued", "Running", "Done"]

            if status == "Queued":
                status = qiskit.providers.jobstatus.JobStatus.QUEUED
            elif status == "Running":
                status = qiskit.providers.jobstatus.JobStatus.RUNNING
            else:
                status = qiskit.providers.jobstatus.JobStatus.DONE
        else:
            status = qiskit.providers.jobstatus.JobStatus.DONE
        return status

    def submit(self) -> None:
        raise NotImplementedError("Submit through SuperstaQBackend, not through SuperstaqJob")
