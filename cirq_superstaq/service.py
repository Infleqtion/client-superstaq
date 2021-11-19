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
from typing import Any, List, Optional, Union

import applications_superstaq
import cirq
import numpy as np
from applications_superstaq import finance
from applications_superstaq import logistics
from applications_superstaq import superstaq_client
from applications_superstaq import user_config

import cirq_superstaq
from cirq_superstaq import job


def counts_to_results(
    counter: collections.Counter, circuit: cirq.AbstractCircuit, param_resolver: cirq.ParamResolver
) -> cirq.Result:
    """Converts a collections.Counter to a cirq.Result.

    Args:
            counter: The collections.Counter of counts for the run.
            circuit: The circuit to run.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in `circuit`.

        Returns:
            A `cirq.Result` for the given circuit and counter.

    """

    measurement_key_names = list(circuit.all_measurement_key_names())
    measurement_key_names.sort()
    # Combines all the measurement key names into a string: {'0', '1'} -> "01"
    combine_key_names = "".join(measurement_key_names)

    samples: List[List[int]] = []
    for key in counter.keys():
        keys_as_list: List[int] = []

        # Combines the keys of the counter into a list. If key = "01", keys_as_list = [0, 1]
        for index in key:
            keys_as_list.append(int(index))

        # Gets the number of counts of the key
        # counter = collections.Counter({"01": 48, "11": 52})["01"] -> 48
        counts_of_key = counter[key]

        # Appends all the keys onto 'samples' list number-of-counts-in-the-key times
        # If collections.Counter({"01": 48, "11": 52}), [0, 1] is appended to 'samples` 48 times and
        # [1, 1] is appended to 'samples' 52 times
        for key in range(counts_of_key):
            samples.append(keys_as_list)

    result = cirq.Result(
        params=param_resolver,
        measurements={
            combine_key_names: np.array(samples),
        },
    )

    return result


class Service(finance.Finance, logistics.Logistics, user_config.UserConfig):
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
            api_version: Version of the api.
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
            client_name="cirq-superstaq",
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def get_counts(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
    ) -> collections.Counter:
        """Runs the given circuit on the SuperstaQ API and returns the result
        of the ran circuit as a collections.Counter

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `collection.Counter` for running the circuit.
        """
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
        counts = self.create_job(resolved_circuit, repetitions, name, target).counts()

        return counts

    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
    ) -> cirq.Result:
        """Run the given circuit on the SuperstaQ API and returns the result
        of the ran circut as a cirq.Result.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        counts = self.get_counts(circuit, repetitions, name, target, param_resolver)
        return counts_to_results(counts, circuit, param_resolver)

    def sampler(self, target: str) -> cirq.Sampler:
        """Returns a `cirq.Sampler` object for accessing sampler interface.

        Args:
            target: Backend to sample against.

        Returns:
            A `cirq.Sampler` for the SuperstaQ API.
        """
        return cirq_superstaq.sampler.Sampler(service=self, target=target)

    def create_job(
        self,
        circuit: cirq.AbstractCircuit,
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
        serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuit)
        result = self._client.create_job(
            serialized_circuits={"cirq_circuits": serialized_circuits},
            repetitions=repetitions,
            target=target,
            name=name,
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result["job_ids"][0])

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

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
        """Get the querying user's account balance in USD.

        Args:
            pretty_output: whether to return a pretty string or a float of the balance.

        Returns:
            If pretty_output is True, returns the balance as a nicely formatted string ($-prefix,
                commas on LHS every three digits, and two digits after period). Otherwise, simply
                returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"${balance:,.2f}"
        return balance

    def get_backends(self) -> dict:
        """Get list of available backends."""
        return self._client.get_backends()["superstaq_backends"]

    def aqt_compile(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]], target: str = "keysight"
    ) -> "cirq_superstaq.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to given target AQT device, optimized to its native gate set.

        Args:
            circuits: cirq Circuit(s) with operations on qubits 4 through 8.
            target: string of target backend AQT device.
        Returns:
            object whose .circuit(s) attribute is an optimized cirq Circuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized cirq.Circuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, cirq.Circuit)

        json_dict = self._client.aqt_compile(
            {"cirq_circuits": serialized_circuits, "backend": target}
        )

        from cirq_superstaq import compiler_output

        return compiler_output.read_json_aqt(json_dict, circuits_list)

    def qscout_compile(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]], target: str = "qscout"
    ) -> "cirq_superstaq.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to given target  QSCOUT device, optimized to its native gate set.

        Args:
            circuits: cirq Circuit(s) with operations on qubits 0 and 1.
            target: string of target backend QSCOUT device.
        Returns:
            object whose .circuit(s) attribute is an optimized cirq Circuit(s)
            and a list of jaqal programs represented as strings
        """
        serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, cirq.Circuit)

        json_dict = self._client.qscout_compile(
            {"cirq_circuits": serialized_circuits, "backend": target}
        )

        from cirq_superstaq import compiler_output

        return compiler_output.read_json_qscout(json_dict, circuits_list)

    def ibmq_compile(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]], target: str = "ibmq_qasm_simulator"
    ) -> Any:
        """Returns pulse schedule for the given circuit and target.

        Qiskit must be installed for returned object to correctly deserialize to a pulse schedule.
        """
        serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)

        json_dict = self._client.ibmq_compile(
            {"cirq_circuits": serialized_circuits, "backend": target}
        )
        try:
            pulses = applications_superstaq.converters.deserialize(json_dict["pulses"])
        except ModuleNotFoundError as e:
            raise applications_superstaq.SuperstaQModuleNotFoundException(
                name=str(e.name), context="ibmq_compile"
            )

        if isinstance(circuits, cirq.Circuit):
            return pulses[0]
        return pulses

    def neutral_atom_compile(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]], target: str = "neutral_atom_qpu"
    ) -> Any:
        """Returns pulse schedule for the given circuit and target.

        Pulse must be installed for returned object to correctly deserialize to a pulse schedule.
        """
        serialized_circuits = cirq_superstaq.serialization.serialize_circuits(circuits)

        json_dict = self._client.neutral_atom_compile(
            {"cirq_circuits": serialized_circuits, "backend": target}
        )
        try:
            pulses = applications_superstaq.converters.deserialize(json_dict["pulses"])
        except ModuleNotFoundError as e:
            raise applications_superstaq.SuperstaQModuleNotFoundException(
                name=str(e.name), context="neutral_atom_compile"
            )

        if isinstance(circuits, cirq.Circuit):
            return pulses[0]
        return pulses
