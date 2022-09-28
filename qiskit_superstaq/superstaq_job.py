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
import requests

import qiskit_superstaq as qss


class SuperstaQJob(qiskit.providers.JobV1):
    def __init__(
        self,
        backend: qss.SuperstaQBackend,
        job_id: str,
    ) -> None:

        # Can we stop setting qobj and access_token to None
        """Initialize a job instance.

        Parameters:
            backend (BaseBackend): Backend that job was executed on.
            job_id (str): The unique job ID from SuperstaQ.
            access_token (str): The access token.
        """
        super().__init__(backend, job_id)

    def __eq__(self, other: Any) -> bool:

        if not (isinstance(other, SuperstaQJob)):
            return False

        return self._job_id == other._job_id

    def _wait_for_results(self, timeout: Optional[float] = None, wait: float = 5) -> List[Dict]:

        result_list: List[Dict] = []
        job_ids = self._job_id.split(",")  # separate aggregated job_ids

        for jid in job_ids:
            start_time = time.time()
            result = None

            while True:
                elapsed = time.time() - start_time

                if timeout and elapsed >= timeout:
                    raise qiskit.providers.JobTimeoutError(
                        "Timed out waiting for result"
                    )  # pragma: no cover b/c don't want slow test or mocking time

                getstr = f"{self._backend.remote_host}/{gss.API_VERSION}/job/{jid}"
                result = requests.get(
                    getstr,
                    headers=self._backend._provider._http_headers(),
                    verify=(self._backend.remote_host == gss.API_URL),
                ).json()

                if result["status"] == "Done":
                    break
                if result["status"] == "Error":
                    raise qiskit.providers.JobError("API returned error:\n" + str(result))
                time.sleep(wait)  # pragma: no cover b/c don't want slow test or mocking time

            result_list.append(result)

        return result_list

    def result(self, timeout: Optional[float] = None, wait: float = 5) -> qiskit.result.Result:
        # Get the result data of a circuit.
        results = self._wait_for_results(timeout, wait)

        # create list of result dictionaries
        results_list = []
        for result in results:
            results_list.append(
                {"success": True, "shots": result["shots"], "data": {"counts": result["samples"]}}
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
        for job_id in job_id_list:
            get_url = f"{self._backend.remote_host}/{gss.API_VERSION}/job/{job_id}"
            result = requests.get(
                get_url,
                headers=self._backend._provider._http_headers(),
                verify=(self._backend.remote_host == gss.API_URL),
            )

            temp_status = result.json()["status"]

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
        return status

    def submit(self) -> None:
        raise NotImplementedError("Submit through SuperstaQBackend, not through SuperstaqJob")
