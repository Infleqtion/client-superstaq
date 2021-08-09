# Copyright 2021 The Cirq Developers
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
"""Service to access SuperstaQs API."""

import collections
import os
from typing import Optional

import cirq

import cirq_superstaq
from cirq_superstaq import job, superstaq_client


class Service:
    """A class to access SuperstaQ's API.

    To access the API, this class requires a remote host url and an API key. These can be
    specified in the constructor via the parameters `remote_host` and `api_key`. Alternatively
    these can be specified by setting the environment variables `SUPERSTAQ_REMOTE_HOST` and
    `SUPERSTAQ_API_KEY`.
    """

    def __init__(
        self,
        remote_host: Optional[str] = None,
        api_key: Optional[str] = None,
        default_target: str = None,
        api_version: str = cirq_superstaq.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
        ibmq_token: str = None,
        ibmq_group: str = None,
        ibmq_project: str = None,
        ibmq_hub: str = None,
        ibmq_pulse: bool = True,
    ):
        """Creates the Service to access SuperstaQ's API.

        Args:
            remote_host: The location of the api in the form of an url. If this is None,
                then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
                If that variable is not set, then this uses
                `flask-service.cgvd1267imk10.us-east-1.cs.amazonlightsail.com/{api_version}`,
                where `{api_version}` is the `api_version` specified below.
            api_key: A string key which allows access to the api. If this is None,
                then this instance will use the environment variable  `SUPERSTAQ_API_KEY`. If that
                variable is not set, then this will raise an `EnvironmentError`.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the api. Defaults to 'v0.1'.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            EnvironmentError: if the `api_key` is None and has no corresponding environment
                variable set.
        """
        self.remote_host = (
            remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or cirq_superstaq.API_URL
        )
        self.api_key = api_key or os.getenv("SUPERSTAQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Parameter api_key was not specified and the environment variable "
                "SUPERSTAQ_API_KEY was also not set."
            )
        self._client = superstaq_client._SuperstaQClient(
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
            ibmq_token=ibmq_token,
            ibmq_group=ibmq_group,
            ibmq_project=ibmq_project,
            ibmq_hub=ibmq_hub,
            ibmq_pulse=ibmq_pulse,
        )

    def run(
        self,
        circuit: "cirq.Circuit",
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
    ) -> collections.Counter:
        """Run the given circuit on the SuperstaQ API.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
        counts = self.create_job(resolved_circuit, repetitions, name, target).counts()

        return counts

    def create_job(
        self,
        circuit: cirq.Circuit,
        repetitions: int = 1000,
        name: Optional[str] = None,
        target: Optional[str] = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.

        Returns:
            A `cirq_superstaq.Job` which can be queried for status or results.

        Raises:
            SuperstaQException: If there was an error accessing the API.
        """
        serialized_program = cirq.to_json(circuit)
        result = self._client.create_job(
            serialized_program=serialized_program, repetitions=repetitions, target=target, name=name
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result["job_id"])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on the SuperstaQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `cirq_superstaq.Job` which can be queried for status or results.

        Raises:
            SuperstaQNotFoundException: If there was no job with the given `job_id`.
            SuperstaQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)

    def aqt_compile(self, circuit: cirq.Circuit) -> "cirq_superstaq.aqt.AQTCompilerOutput":
        """Compiles the given circuit to AQT device, optimized to its native gate set.

        Args:
            circuit: a cirq Circuit object with operations on qubits 4 through 8.
        Returns:
            AQTCompilerOutput object, whose .circuit attribute contains an optimized cirq Circuit.
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized cirq Circuit.
        """
        serialized_program = cirq.to_json(circuit)
        json_dict = self._client.aqt_compile(serialized_program)
        from cirq_superstaq import aqt

        return aqt.read_json(json_dict)
