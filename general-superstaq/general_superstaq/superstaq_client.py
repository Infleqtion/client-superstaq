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
"""Client for making requests to Superstaq's API."""

from __future__ import annotations

import copy
import enum
import json
import os
import pathlib
import sys
import textwrap
import time
import urllib
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeVar

import requests

import general_superstaq as gss
from general_superstaq import _models

TQuboKey = TypeVar("TQuboKey")

RECOGNISED_CIRCUIT_TYPES = Literal[_models.CircuitType.CIRQ, _models.CircuitType.QISKIT]
"""The circuit types that are currently implemented within the SuperstaqClient."""


class ApiVersion(str, enum.Enum):
    """The supported API versions."""

    V0_2_0 = "v0.2.0"
    V0_3_0 = "v0.3.0"


class _BaseSuperstaqClient(ABC):
    """Handles calls to Superstaq's API.

    Users should not instantiate this themselves,
    but instead should use `$client_superstaq.Service`.
    """

    RETRIABLE_STATUS_CODES: ClassVar[set[int]] = {
        requests.codes.service_unavailable,
    }
    SUPPORTED_VERSIONS: ClassVar[set[str]] = {v.value for v in ApiVersion}

    def __init__(
        self,
        client_name: str,
        api_key: str | None = None,
        remote_host: str | None = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
        cq_token: str | None = None,
        ibmq_token: str | None = None,
        ibmq_instance: str | None = None,
        ibmq_channel: str | None = None,
        use_stored_ibmq_credentials: bool = False,
        ibmq_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the SuperstaqClient.

        Users should use `$client_superstaq.Service` instead of this class directly.

        The SuperstaqClient handles making requests to the SuperstaqClient,
        returning dictionary results. It handles retry and authentication.

        Args:
            client_name: The name of the client.
            api_key: The key used for authenticating against the Superstaq API.
            remote_host: The url of the server exposing the Superstaq API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_version: Which version of the API to use. Defaults to `client_superstaq.API_VERSION`
                (which is the most recent version when this client was downloaded).
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
            cq_token: Token from CQ cloud. This may be required to submit circuits to CQ hardware.
            ibmq_token: An optional IBM Quantum or IBM Cloud token. This may be required to submit
                circuits to IBM hardware, or to access non-public IBM devices you may have access
                to.
            ibmq_instance: An optional instance to use when running IBM jobs.
            ibmq_channel: Optional type of IBM account. Must be either "ibm_quantum_platform" or
                "ibm_cloud".
            use_stored_ibmq_credentials: Boolean flag on whether to retrieve IBM credentials from
                locally saved accounts or not. Defaults to `False`.
            ibmq_name: The name of the account to retrieve. The default is `default-ibm-quantum`.
            kwargs: Other optimization and execution parameters.
        """
        self.api_key = api_key or gss.superstaq_client.find_api_key()
        self.remote_host = remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or gss.API_URL
        self.client_name = client_name
        self.api_version = api_version
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        url = urllib.parse.urlparse(self.remote_host)
        assert url.scheme, (
            f"Specified URL protocol/scheme in `remote_host` ({self.remote_host}) is not valid. "
            "Use, for example, 'http', 'https'."
        )
        assert url.netloc, (
            f"Specified network location in `remote_host` ({self.remote_host}) is not a valid URL "
            "like, for example, http://example.com"
        )

        assert self.api_version in self.SUPPORTED_VERSIONS, (
            f"Only API versions {self.SUPPORTED_VERSIONS} are accepted but got {self.api_version}"
        )
        assert max_retry_seconds >= 0, "Negative retry not possible without time machine."

        self.url = f"{url.scheme}://{url.netloc}/{self.api_version}"
        self.verify_https: bool = f"{gss.API_URL}/{self.api_version}" == self.url
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "X-Client-Name": self.client_name,
            "X-Client-Version": self.api_version,
        }
        self.session = requests.Session()

        if cq_token:
            kwargs["cq_token"] = cq_token

        if use_stored_ibmq_credentials:
            config = read_ibm_credentials(ibmq_name)
            ibmq_token = config.get("token")
            ibmq_instance = config.get("instance")
            ibmq_channel = config.get("channel")

        if ibmq_channel:
            gss.validation._validate_ibm_channel(ibmq_channel)

        if ibmq_token:
            kwargs["ibmq_token"] = ibmq_token
        if ibmq_instance:
            kwargs["ibmq_instance"] = ibmq_instance
        if ibmq_channel:
            kwargs["ibmq_channel"] = ibmq_channel

        self.client_kwargs = kwargs

        if self.api_version == "v0.3.0":
            self._warn_unstable_version()

    def get_superstaq_version(self) -> dict[str, str | None]:
        """Gets Superstaq version from response header.

        Returns:
            A `dict` containing the current Superstaq version.
        """
        response = self.session.get(self.url)
        version = response.headers.get("superstaq_version")

        return {"superstaq_version": version}

    @abstractmethod
    def create_job(  # type: ignore [return]
        self,
        serialized_circuits: dict[str, str],
        repetitions: int = 1,
        target: str = "ss_unconstrained_simulator",
        method: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a job.

        Args:
            serialized_circuits: The serialized representation of the circuit to run.
            repetitions: The number of times to repeat the circuit. For simulation the repeated
                sampling is not done on the server, but is passed as metadata to be recovered
                from the returned job.
            target: Target to run on.
            method: Which type of method to execute the circuits (noisy simulator,
            non-noisy simulator, hardware, e.t.c)
            kwargs: Other optimization and execution parameters.

        Returns:
            The json body of the response as a dict. This does not contain populated information
            about the job, but does contain the job id.

        Raises:
            ~gss.SuperstaqServerException: if the request fails.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_integer_param(repetitions, min_val=1)

    @abstractmethod
    def cancel_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> list[str]:
        """Cancel jobs associated with given job ids.

        Args:
            job_ids: The UUIDs of the jobs (returned when the jobs were created).
            kwargs: Extra options needed to fetch jobs.

        Returns:
            A list of the job ids of the jobs that successfully cancelled.

        Raises:
            ~gss.SuperstaqServerException: For other API call failures.
        """

    @abstractmethod
    def fetch_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> dict[str, dict[str, str]]:
        """Get the job from the Superstaq API.

        Args:
            job_ids: The UUIDs of the jobs (returned when the jobs were created).
            kwargs:  Extra options needed to fetch jobs.

        Returns:
            The json body of the response as a dict.

        Raises:
            ~gss.SuperstaqServerException: For other API call failures.
        """

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Get the querying user's account balance in USD.

        Returns:
            The json body of the response as a dict.
        """

    @abstractmethod
    def get_user_info(
        self,
        name: str | None = None,
        email: str | None = None,
        user_id: int | uuid.UUID | None = None,
    ) -> list[dict[str, str | float]]:
        """Gets a dictionary of the user's info.

        .. note::

            SUPERTECH users can submit optional :code:`name` or :code:`email`
            arguments which can be used to search for the info of arbitrary users on the server.

        Args:
            name: A name to search by. Defaults to None.
            email: An email address to search by. Defaults to None.
            user_id: A user ID to search by. Defaults to None.

        Returns:
            A list of dictionaries corresponding to the user
            information for each user that matches the query. If no :code:`name` or :code:`email`
            parameters are used this dictionary will have length 1.

        Raises:
            ~gss.SuperstaqServerException: If the server returns an empty response.
        """

    @abstractmethod
    def _accept_terms_of_use(self, user_input: str) -> str:
        """Makes a request to Superstaq API to confirm acceptance of terms of use.

        Args:
            user_input: The user's response to prompt for acceptance of TOU. Server accepts YES.

        Returns:
            String with success message.
        """

    @abstractmethod
    def get_targets(self, **kwargs: bool | None) -> list[gss.Target]:
        """Makes a GET request to retrieve targets from the Superstaq API.

        Args:
            kwargs: Optional flags to restrict/filter returned targets.

        Returns:
            A list of Superstaq targets matching all provided criteria.
        """

    @abstractmethod
    def target_info(self, target: str, **kwargs: object) -> dict[str, Any]:  # type: ignore [return]
        """Makes a POST request to the /target_info endpoint.

        Uses the Superstaq API to request information about `target`.

        Args:
            target: A string representing the device to get information about.
            kwargs: Any other information.

        Returns:
            The target information.
        """
        gss.validation.validate_target(target)

    @abstractmethod
    def get_my_targets(self) -> list[gss.Target]:
        """Makes a GET request to retrieve targets from the Superstaq API.

        Returns:
            A list of Superstaq targets matching all provided criteria.
        """

    @abstractmethod
    def add_new_user(self, json_dict: dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to add a new user.

        Args:
            json_dict: The dictionary with user entry.

        Returns:
            The response as a string.
        """

    @abstractmethod
    def update_user_balance(self, json_dict: dict[str, float | str]) -> str:
        """Makes a POST request to Superstaq API to update a user's balance in the database.

        Args:
            json_dict: The dictionary with user entry and new balance.

        Returns:
            The response as a string.
        """

    @abstractmethod
    def update_user_role(self, json_dict: dict[str, int | str]) -> str:
        """Makes a POST request to Superstaq API to update a user's role.

        Args:
            json_dict: The dictionary with user entry and new role.

        Returns:
            The response as a string.
        """

    @abstractmethod
    def resource_estimate(self, json_dict: dict[str, str]) -> dict[str, list[dict[str, int]]]:
        """POSTs the given payload to the `/resource_estimate` endpoint.

        Args:
            json_dict: The payload to POST.

        Returns:
            The response of the given payload.
        """

    @abstractmethod
    def aqt_compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        """Makes a POST request to Superstaq API to compile a list of circuits for Berkeley-AQT.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit(s) data.
        """

    @abstractmethod
    def qscout_compile(self, json_dict: dict[str, str]) -> dict[str, str | list[str]]:
        """Makes a POST request to Superstaq API to compile a list of circuits for QSCOUT.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit(s) data.
        """

    @abstractmethod
    def compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        """Makes a POST request to Superstaq API to compile a list of circuits.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit data.
        """

    @abstractmethod
    def submit_qubo(  # type: ignore [return]
        self,
        qubo: Mapping[tuple[TQuboKey, ...], float],
        target: str,
        repetitions: int,
        method: str = "sim_anneal",
        max_solutions: int | None = 1000,
        *,
        qaoa_depth: int = 1,
        rqaoa_cutoff: int = 0,
        dry_run: bool = False,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> dict[str, str]:
        """Makes a POST request to Superstaq API to submit a QUBO problem to the
        given target.

        Args:
            qubo: A dictionary representing the QUBO object. The tuple keys represent the
                boolean variables of the QUBO and the values represent the coefficients.
                As an example, for a QUBO with integer coefficients = 2*a + a*b - 5*b*c - 3
                (where a, b, and c are boolean variables), the corresponding dictionary format
                would be {('a',): 2, ('a', 'b'): 1, ('b', 'c'): -5, (): -3}.
            target: The target to submit the QUBO.
            repetitions: Number of times that the execution is repeated before stopping.
            method: The parameter specifying method of QUBO solving execution. Currently, the
                supported methods include "bruteforce", "sim_anneal", "qaoa", or "rqaoa".
                Defaults to "sim_anneal" which runs on DWave's simulated annealer.
            max_solutions: A parameter that specifies the max number of output solutions.
            qaoa_depth: The number of QAOA layers to use. Defaults to 1.
            rqaoa_cutoff: The stopping point for RQAOA before using switching to a classical
                solver. Defaults to 0.
            dry_run: If `method="qaoa"`, a boolean flag to (not) run an ideal 'dry-run'
                QAOA execution on `target`.
            random_seed: Optional random seed choice for RQAOA.
            kwargs: Any additional keyword arguments supported by the qubo solver methods.

        Returns:
            A dictionary from the POST request.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_qubo(qubo)
        gss.validation.validate_integer_param(repetitions)
        gss.validation.validate_integer_param(max_solutions)
        gss.validation.validate_integer_param(qaoa_depth)
        gss.validation.validate_integer_param(rqaoa_cutoff, min_val=0)

    @abstractmethod
    def supercheq(
        self,
        files: list[list[int]],
        num_qubits: int,
        depth: int,
        circuit_return_type: str,
    ) -> Any:
        """Performs a POST request on the `/supercheq` endpoint.

        Args:
            files: List of files specified as binary using integers.
                For example: [[1, 0, 1], [1, 1, 1]].
            num_qubits: Number of qubits to run Supercheq on.
            depth: The depth of the circuits to run Supercheq on.
            circuit_return_type: Supports only `cirq` and `qiskit` for now.

        Returns:
            The output of Supercheq.
        """
        gss.validation.validate_integer_param(num_qubits, min_val=1)
        gss.validation.validate_integer_param(depth, min_val=1)

    @abstractmethod
    def submit_dfe(  # type: ignore [return]
        self,
        circuit_1: dict[str, str],
        target_1: str,
        circuit_2: dict[str, str],
        target_2: str,
        num_random_bases: int,
        shots: int,
        **kwargs: Any,
    ) -> list[str]:
        """Performs a POST request on the `/dfe_post` endpoint.

        Args:
            circuit_1: Serialized circuit that prepares the first state for the protocol.
            target_1: Target to prepare the first state on.
            circuit_2: Serialized circuit that prepares the second state for the protocol.
            target_2: Target to prepare the second state on.
            num_random_bases: Number of random bases to measure the states on.
            shots: Number of shots per random basis.
            kwargs: Other execution parameters.
                - tag: Tag for all jobs submitted for this protocol.
                - lifespan: How long to store the jobs submitted for in days (only works with right
                permissions).
                - method: Which type of method to execute the circuits with.

        Returns:
            A list of size two with the ids for the RMT jobs created; these ids should be passed to
            `process_dfe` to get back the fidelity estimation.

        Raises:
            ValueError: If any of the targets passed are not valid.
            ~gss.SuperstaqServerException: if the request fails.
        """
        gss.validation.validate_target(target_1)
        gss.validation.validate_target(target_2)
        gss.validation.validate_integer_param(num_random_bases, min_val=1)
        gss.validation.validate_integer_param(shots, min_val=1)

    @abstractmethod
    def process_dfe(self, job_ids: Sequence[str] | Sequence[uuid.UUID]) -> float:
        """Performs a POST request on the `/dfe_fetch` endpoint.

        Args:
            job_ids: A list of job ids returned by a call to `submit_dfe`.

        Returns:
            The estimated fidelity between the two states as a float.

        Raises:
            ValueError: If `job_ids` is not of size two.
            ~gss.SuperstaqServerException: If the request fails.
        """

    @abstractmethod
    def submit_aces(  # type: ignore [return]
        self,
        target: str,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: str | None = None,
        noise: dict[str, object] | None = None,
        tag: str | None = None,
        lifespan: int | None = None,
        weights: Sequence[int] | None = None,
    ) -> str:
        """Performs a POST request on the `/aces` endpoint.

        Args:
            target: The device target to characterize.
            qubits: A list with the qubit indices to characterize.
            shots: How many shots to use per circuit submitted.
            num_circuits: How many random circuits to use in the protocol.
            mirror_depth: The half-depth of the mirror portion of the random circuits.
            extra_depth: The depth of the fully random portion of the random circuits.
            method: Which type of method to execute the circuits with.
            noise: A dictionary describing a noise model to simulate the run with.
            tag: Tag for all jobs submitted for this protocol.
            lifespan: How long to store the jobs submitted for in days (only works with right
                permissions).
            weights: The weights of the Pauli strings.

        Returns:
            A string with the job id for the ACES job created.

        Raises:
            ValueError: If the target or noise model is not valid.
            ~gss.SuperstaqServerException: If the request fails.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_integer_param(shots, min_val=1)
        gss.validation.validate_integer_param(num_circuits, min_val=1)
        gss.validation.validate_integer_param(mirror_depth, min_val=1)
        gss.validation.validate_integer_param(extra_depth, min_val=0)

    @abstractmethod
    def process_aces(self, job_id: str | uuid.UUID) -> list[float]:
        """Makes a POST request to the "/aces_fetch" endpoint.

        Args:
            job_id: The job id returned by `submit_aces`.

        Returns:
            The estimated eigenvalues.
        """

    @abstractmethod
    def submit_cb(  # type: ignore [return]
        self,
        target: str,
        shots: int,
        serialized_circuits: dict[str, str],
        n_channels: int,
        n_sequences: int,
        depths: Sequence[int],
        method: str | None = None,
        noise: dict[str, object] | None = None,
    ) -> str:
        """Makes a POST request to the `/cycle_benchmarking` endpoint.

        Args:
            target: The target device to characterize.
            shots: How many shots to use per circuit submitted.
            serialized_circuits: The serialized process circuits to use in the protocol.
            n_channels: The number of random Pauli decay channels to approximate error.
            n_sequences: Number of circuits to generate per depth.
            depths: Lists of depths representing the depths of Cycle Benchmarking circuits
                to generate.
            method: Optional method to use in device submission (e.g. "dry-run").
            noise: Dictionary representing noise model to simulate the protocol with.

        Returns:
            A string with the job id for the Cycle Benchmarking job created.

        Raises:
            ValueError: If the target or noise model is not valid.
            ~gss.SuperstaqServerException: If the request fails.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_integer_param(shots, min_val=1)
        gss.validation.validate_integer_param(n_channels, min_val=1)
        gss.validation.validate_integer_param(n_sequences, min_val=1)

    @abstractmethod
    def process_cb(self, job_id: str | uuid.UUID, counts: str | None = None) -> dict[str, Any]:
        """Makes a POST request to the "/cb_fetch" endpoint.

        Args:
            job_id: The job id returned by `submit_cb`.
            counts: Optional dict representing result counts.

        Returns:
            Characterization data including process fidelity
            and parameter estimates.
        """

    @abstractmethod
    def aqt_upload_configs(self, aqt_configs: dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to upload configurations.

        Args:
            aqt_configs: The configs to be uploaded.

        Returns:
            A string response from POST request.
        """

    @abstractmethod
    def aqt_get_configs(self) -> dict[str, str]:
        """Writes AQT configs from the AQT system onto the given file path.

        Returns:
            A dictionary containing the AQT configs.
        """

    def get_request(
        self, endpoint: str, query: Mapping[str, object] | None = None, **credentials: str
    ) -> Any:
        """Performs a GET request on a given endpoint.

        Args:
            endpoint: The endpoint to perform the GET request on.
            query: An optional query dictionary to include in the get request.
                This query will be appended to the url.
            credentials: Any credentials that need to be added to the request headers.

        Returns:
            The response of the GET request.
        """

        def request() -> requests.Response:
            """Builds GET request object.

            Returns:
                The Flask GET request object.
            """
            if not query:
                q_string = ""
            else:
                q_string = "?" + urllib.parse.urlencode(query, doseq=True)
            return self.session.get(
                f"{self.url}{endpoint}{q_string}",
                headers=self._custom_headers(**credentials),
                verify=self.verify_https,
            )

        response = self._make_request(request)
        return self._handle_response(response)

    def post_request(
        self, endpoint: str, json_dict: Mapping[str, object], **credentials: str
    ) -> Any:
        """Performs a POST request on a given endpoint with a given payload.

        Args:
            endpoint: The endpoint to perform the POST request on.
            json_dict: The payload to POST.
            credentials: Any credentials that need to be added to the request headers.

        Returns:
            The response of the POST request.
        """

        def request() -> requests.Response:
            """Builds GET request object.

            Returns:
                The Flask GET request object.
            """
            return self.session.post(
                f"{self.url}{endpoint}",
                json=json_dict,
                headers=self._custom_headers(**credentials),
                verify=self.verify_https,
            )

        response = self._make_request(request)
        return self._handle_response(response)

    def put_request(
        self, endpoint: str, json_dict: Mapping[str, object], **credentials: str
    ) -> Any:
        """Performs a PUT request on a given endpoint with a given payload.

        Args:
            endpoint: The endpoint to perform the PUT request on.
            json_dict: The payload to PUT.
            credentials: Any credentials that need to be added to the request headers.

        Returns:
            The response of the PUT request.
        """

        # Update headers with custom values
        def request() -> requests.Response:
            """Builds PUT request object.

            Returns:
                The Flask PUT request object.
            """
            return self.session.put(
                f"{self.url}{endpoint}",
                json=json_dict,
                headers=self._custom_headers(**credentials),
                verify=self.verify_https,
            )

        response = self._make_request(request)
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> object:
        response_json = response.json()
        if isinstance(response_json, dict) and "warnings" in response_json:
            for warning in response_json["warnings"]:
                warnings.warn(warning["message"], gss.SuperstaqWarning, stacklevel=4)
            del response_json["warnings"]
        return response_json

    def _handle_status_codes(self, response: requests.Response) -> None:
        """A method to handle status codes.

        Args:
            response: The `requests.Response` to get the status codes from.

        Raises:
            ~gss.SuperstaqServerException: If unauthorized by Superstaq API.
            ~gss.SuperstaqServerException: If an error has occurred in making a request
                to the Superstaq API.
        """
        if response.status_code == requests.codes.unauthorized:
            if response.json() == (
                "You must accept the Terms of Use (superstaq.infleqtion.com/terms_of_use)."
            ):
                self._prompt_accept_terms_of_use()
                return

            elif response.json() == ("You must validate your registered email."):
                raise gss.SuperstaqServerException(
                    "You must validate your registered email.",
                    response.status_code,
                )

            else:
                raise gss.SuperstaqServerException(
                    '"Not authorized" returned by Superstaq API.  '
                    "Check to ensure you have supplied the correct API key.",
                    response.status_code,
                )

        if response.status_code == requests.codes.gateway_timeout:
            # Job took too long. Don't retry, it probably won't be any faster.
            raise gss.SuperstaqServerException(
                "Connection timed out while processing your request. Try submitting a smaller "
                "batch of circuits.",
                response.status_code,
            )

        if response.status_code not in self.RETRIABLE_STATUS_CODES:
            try:
                json_content = self._handle_response(response)
            except requests.JSONDecodeError:
                json_content = None

            if isinstance(json_content, dict) and set(json_content.keys()).intersection(
                {"message", "details"}
            ):
                alternative: str = json_content.get("details", "")
                message: str = json_content.get("message", alternative)
            else:
                message = str(response.text)

            raise gss.SuperstaqServerException(
                message=message, status_code=response.status_code, contact_info=True
            )

    def _make_request(self, request: Callable[[], requests.Response]) -> requests.Response:
        """Make a request to the API, retrying if necessary.

        Args:
            request: A function that returns a `requests.Response`.

        Raises:
            ~gss.SuperstaqServerException: If there was a not-retriable error from
                the API.
            TimeoutError: If the requests retried for more than `max_retry_seconds`.

        Returns:
            The `requests.Response` from the final successful request call.
        """
        # Initial backoff of 100ms.
        delay_seconds = 0.1
        while True:
            try:
                response = request()

                if response.ok:
                    return response

                self._handle_status_codes(response)
                message = response.reason

            # Fallthrough should retry.
            except requests.RequestException as e:
                # Connection error, timeout at server, or too many redirects.
                # Retry these.
                message = f"RequestException of type {type(e)}."
            if delay_seconds > self.max_retry_seconds:
                raise TimeoutError(f"Reached maximum number of retries. Last error: {message}")
            if self.verbose:
                print(message, file=sys.stderr)  # noqa: T201
                print(f"Waiting {delay_seconds} seconds before retrying.")  # noqa: T201
            time.sleep(delay_seconds)
            delay_seconds *= 2

    def _warn_unstable_version(self) -> None:
        warnings.warn(
            (
                "Version v0.3.0 of the Superstaq endpoint is currently under development. "
                "This version should not be treated as stable. Most user should continue using "
                "`api_version='v0.2.0'`."
            ),
            stacklevel=2,
        )

    def _raise_not_implemented(self, function_name: str) -> None:
        raise NotImplementedError(
            f"The function {function_name} is not implemented for version {self.api_version}."
        )

    def _custom_headers(self, **credentials: str) -> dict[str, str]:
        custom_headers = copy.deepcopy(self.headers)
        for key in ["ibmq_token", "ibmq_instance", "ibmq_channel", "cq_token"]:
            if key in credentials:
                custom_headers[key] = credentials[key]

        return custom_headers

    def _prompt_accept_terms_of_use(self) -> None:
        """Prompts terms of use.

        Raises:
            ~gss.SuperstaqServerException: If terms of use are not accepted.
        """
        message = (
            "Acceptance of the Terms of Use (superstaq.infleqtion.com/terms_of_use)"
            " is necessary before using Superstaq.\nType in YES to accept: "
        )
        user_input = input(message).upper()
        response = self._accept_terms_of_use(user_input)
        print(response)  # noqa: T201
        if response != "Accepted. You can now continue using Superstaq.":
            raise gss.SuperstaqServerException(
                "You'll need to accept the Terms of Use before usage of Superstaq.",
                requests.codes.unauthorized,
            )

    @staticmethod
    def _extract_credentials(kwargs: dict[str, Any]) -> dict[str, str]:
        credentials = {}
        for key in ["ibmq_token", "ibmq_instance", "ibmq_channel"]:
            if key in kwargs:
                credentials[key] = kwargs.pop(key)

        if "cq_token" in kwargs:  # CQ-Token may need to be json encoded as headers must be strings.
            cq_token = kwargs.pop("cq_token")
            if isinstance(cq_token, str):
                credentials["cq_token"] = cq_token
            else:
                credentials["cq_token"] = json.dumps(cq_token)

        return credentials

    @staticmethod
    def _extract_circuits(json_dict: dict[str, str]) -> tuple[str, RECOGNISED_CIRCUIT_TYPES]:
        recognised_circuit_types: dict[str, RECOGNISED_CIRCUIT_TYPES] = {
            "cirq_circuits": _models.CircuitType.CIRQ,
            "qiskit_circuits": _models.CircuitType.QISKIT,
        }

        circuit_keys = list(filter(lambda x: x in recognised_circuit_types, json_dict))

        if len(circuit_keys) > 1:
            raise RuntimeError("Cannot submit jobs with multiple circuit types.")

        if len(circuit_keys) == 0:
            raise RuntimeError("No recognized circuits found.")

        circuits = json_dict[circuit_keys[0]]
        circuit_type = recognised_circuit_types[circuit_keys[0]]

        return circuits, circuit_type

    def __str__(self) -> str:
        return f"Client version {self.api_version} with host={self.url} and name={self.client_name}"

    def __repr__(self) -> str:
        return textwrap.dedent(
            f"""\
            gss.superstaq_client._SuperstaqClient(
                remote_host={self.url!r},
                api_key={self.api_key!r},
                client_name={self.client_name!r},
                api_version={self.api_version!r},
                max_retry_seconds={self.max_retry_seconds!r},
                verbose={self.verbose!r},
            )"""
        )


class _SuperstaqClient(_BaseSuperstaqClient):
    def create_job(
        self,
        serialized_circuits: dict[str, str],
        repetitions: int = 1,
        target: str = "ss_unconstrained_simulator",
        method: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        super().create_job(serialized_circuits, repetitions, target)

        json_dict: dict[str, Any] = {
            **serialized_circuits,
            "target": target,
            "shots": int(repetitions),
        }
        if method is not None:
            json_dict["method"] = method
        if kwargs or self.client_kwargs:
            json_dict["options"] = json.dumps({**self.client_kwargs, **kwargs})
        return self.post_request("/jobs", json_dict)

    def cancel_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> list[str]:
        json_dict: dict[str, Any] = {
            "job_ids": job_ids,
        }
        if kwargs or self.client_kwargs:
            json_dict["options"] = json.dumps({**self.client_kwargs, **kwargs})

        return self.post_request("/cancel_jobs", json_dict)["succeeded"]

    def fetch_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> dict[str, dict[str, str]]:
        json_dict: dict[str, Any] = {
            "job_ids": job_ids,
        }
        if kwargs or self.client_kwargs:
            json_dict["options"] = json.dumps({**self.client_kwargs, **kwargs})

        return self.post_request("/fetch_jobs", json_dict)

    def get_balance(self) -> dict[str, float]:
        return self.get_request("/balance")

    def get_user_info(
        self,
        name: str | None = None,
        email: str | None = None,
        user_id: int | uuid.UUID | None = None,
    ) -> list[dict[str, str | float]]:
        query = {}
        if name is not None:
            query["name"] = name
        if email is not None:
            query["email"] = email
        if user_id is not None:
            query["id"] = str(user_id)
        user_info = self.get_request("/user_info", query=query)
        if not user_info:
            # Catch empty server response. This shouldn't happen as the server should return
            # an error code if something is wrong with the request.
            raise gss.SuperstaqServerException(
                "Something went wrong. The server has returned an empty response."
            )
        return list(user_info.values())

    def _accept_terms_of_use(self, user_input: str) -> str:
        return self.post_request("/accept_terms_of_use", {"user_input": user_input})

    def get_targets(self, **kwargs: bool | None) -> list[gss.Target]:
        json_dict: dict[str, str | bool] = {
            key: value for key, value in kwargs.items() if value is not None
        }
        if self.client_kwargs:
            json_dict["options"] = json.dumps(self.client_kwargs)
        superstaq_targets = self.post_request("/targets", json_dict)["superstaq_targets"]
        target_list = [
            gss.Target(target=target_name, **properties)
            for target_name, properties in superstaq_targets.items()
        ]
        return target_list

    def get_my_targets(self) -> list[gss.Target]:
        json_dict: dict[str, str | bool] = {"accessible": True}
        if self.client_kwargs:
            json_dict["options"] = json.dumps(self.client_kwargs)

        superstaq_targets = self.post_request("/targets", json_dict)["superstaq_targets"]
        target_list = [
            gss.Target(target=target_name, **properties)
            for target_name, properties in superstaq_targets.items()
        ]
        return target_list

    def target_info(self, target: str, **kwargs: object) -> dict[str, Any]:
        super().target_info(target)

        json_dict = {
            "target": target,
            "options": json.dumps(self.client_kwargs),
        }
        return self.post_request("/target_info", json_dict)

    def add_new_user(self, json_dict: dict[str, str]) -> str:
        return self.post_request("/add_new_user", json_dict)

    def update_user_balance(self, json_dict: dict[str, float | str]) -> str:
        return self.post_request("/update_user_balance", json_dict)

    def update_user_role(self, json_dict: dict[str, int | str]) -> str:
        return self.post_request("/update_user_role", json_dict)

    def resource_estimate(self, json_dict: dict[str, str]) -> dict[str, list[dict[str, int]]]:
        return self.post_request("/resource_estimate", json_dict)

    def aqt_compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        return self.post_request("/aqt_compile", json_dict)

    def qscout_compile(self, json_dict: dict[str, str]) -> dict[str, str | list[str]]:
        return self.post_request("/qscout_compile", json_dict)

    def compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        return self.post_request("/compile", json_dict)

    def submit_qubo(
        self,
        qubo: Mapping[tuple[TQuboKey, ...], float],
        target: str,
        repetitions: int,
        method: str = "sim_anneal",
        max_solutions: int | None = 1000,
        *,
        qaoa_depth: int = 1,
        rqaoa_cutoff: int = 0,
        dry_run: bool = False,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> dict[str, str]:
        super().submit_qubo(
            qubo,
            target,
            repetitions,
            max_solutions=max_solutions,
            qaoa_depth=qaoa_depth,
            rqaoa_cutoff=rqaoa_cutoff,
        )

        options = {
            "qaoa_depth": qaoa_depth,
            "rqaoa_cutoff": rqaoa_cutoff,
            "random_seed": random_seed,
            **kwargs,
        }

        json_dict = {
            "qubo": list(qubo.items()),
            "target": target,
            "shots": int(repetitions),
            "method": method,
            "max_solutions": max_solutions,
            "dry_run": dry_run,
            "options": json.dumps(options),
        }
        return self.post_request("/qubo", json_dict)

    def supercheq(
        self,
        files: list[list[int]],
        num_qubits: int,
        depth: int,
        circuit_return_type: str,
    ) -> Any:
        super().supercheq(files, num_qubits, depth, circuit_return_type)

        json_dict = {
            "files": files,
            "num_qubits": int(num_qubits),
            "depth": int(depth),
            "circuit_return_type": circuit_return_type,
        }
        return self.post_request("/supercheq", json_dict)

    def submit_dfe(
        self,
        circuit_1: dict[str, str],
        target_1: str,
        circuit_2: dict[str, str],
        target_2: str,
        num_random_bases: int,
        shots: int,
        **kwargs: Any,
    ) -> list[str]:
        super().submit_dfe(circuit_1, target_1, circuit_2, target_2, num_random_bases, shots)

        state_1 = {**circuit_1, "target": target_1}
        state_2 = {**circuit_2, "target": target_2}

        json_dict: dict[str, Any] = {
            "state_1": state_1,
            "state_2": state_2,
            "shots": int(shots),
            "n_bases": int(num_random_bases),
        }

        if kwargs:
            json_dict["options"] = json.dumps(kwargs)
        return self.post_request("/dfe_post", json_dict)

    def process_dfe(self, job_ids: Sequence[str] | Sequence[uuid.UUID]) -> float:
        if len(job_ids) != 2:
            raise ValueError("`job_ids` must contain exactly two job ids.")

        json_dict = {
            "job_id_1": job_ids[0],
            "job_id_2": job_ids[1],
        }
        return self.post_request("/dfe_fetch", json_dict)

    def submit_aces(
        self,
        target: str,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: str | None = None,
        noise: dict[str, object] | None = None,
        tag: str | None = None,
        lifespan: int | None = None,
        weights: Sequence[int] | None = None,
    ) -> str:
        super().submit_aces(
            target,
            qubits,
            shots,
            num_circuits,
            mirror_depth,
            extra_depth,
        )

        json_dict = {
            "target": target,
            "qubits": qubits,
            "shots": shots,
            "num_circuits": num_circuits,
            "mirror_depth": mirror_depth,
            "extra_depth": extra_depth,
        }

        if weights:
            json_dict["weights"] = weights
        if method:
            json_dict["method"] = method
        if noise:
            if "type" in noise.keys():
                gss.validation.validate_noise_type(noise, len(qubits))
            json_dict["noise"] = noise
        if tag:
            json_dict["tag"] = tag
        if lifespan:
            json_dict["lifespan"] = lifespan

        return self.post_request("/aces", json_dict)

    def process_aces(self, job_id: str | uuid.UUID) -> list[float]:
        return self.post_request("/aces_fetch", {"job_id": job_id})

    def submit_cb(
        self,
        target: str,
        shots: int,
        serialized_circuits: dict[str, str],
        n_channels: int,
        n_sequences: int,
        depths: Sequence[int],
        method: str | None = None,
        noise: dict[str, object] | None = None,
    ) -> str:
        super().submit_cb(
            target,
            shots,
            serialized_circuits,
            n_channels,
            n_sequences,
            depths,
        )

        json_dict: dict[str, Any] = {
            "target": target,
            "shots": shots,
            **serialized_circuits,
            "n_channels": n_channels,
            "n_sequences": n_sequences,
            "depths": depths,
        }

        if method:
            json_dict["method"] = method
        if noise:
            json_dict["noise"] = noise

        return self.post_request("/cb_submit", json_dict)

    def process_cb(self, job_id: str | uuid.UUID, counts: str | None = None) -> dict[str, Any]:
        json_dict: dict[str, Any] = {
            "job_id": job_id,
        }
        if counts:
            json_dict["counts"] = counts
        return self.post_request("/cb_fetch", json_dict)

    def aqt_upload_configs(self, aqt_configs: dict[str, str]) -> str:
        return self.post_request("/aqt_configs", aqt_configs)

    def aqt_get_configs(self) -> dict[str, str]:
        return self.get_request("/get_aqt_configs")


class _SuperstaqClientV3(_BaseSuperstaqClient):
    def create_job(
        self,
        serialized_circuits: dict[str, str],
        repetitions: int = 1,
        target: str = "ss_unconstrained_simulator",
        method: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Version 0.3.0 implementation."""
        super().create_job(serialized_circuits, repetitions, target)

        # Infer the job type
        if target.endswith("_simulator") or method in _models.SimMethod._value2member_map_.keys():
            job_type = _models.JobType.SIMULATE
        else:
            job_type = _models.JobType.SUBMIT

        # Get sim method if needed
        if job_type == _models.JobType.SIMULATE:
            if method in _models.SimMethod._value2member_map_.keys():
                sim_method = _models.SimMethod(method)
            else:
                sim_method = None
        else:
            sim_method = None

        # Infer dry run
        dry_run = method == "dry-run"

        # Get circuits
        circuits, circuit_type = self._extract_circuits(serialized_circuits)

        # Extract tokens from kwargs and move to the header
        credentials = self._extract_credentials({**kwargs, **self.client_kwargs})

        new_job = _models.NewJob(
            job_type=job_type,
            target=target,
            circuits=circuits,
            circuit_type=circuit_type,
            dry_run=dry_run,
            sim_method=sim_method,
            shots=repetitions,
            options_dict={**self.client_kwargs, **kwargs},
        )
        response = _models.NewJobResponse(
            **self.post_request("/client/job", new_job.model_dump(), **credentials)
        )
        return response.model_dump()

    def cancel_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> list[str]:
        query = _models.JobQuery(job_id=job_ids)
        credentials = self._extract_credentials({**kwargs, **self.client_kwargs})
        response = _models.JobCancellationResults(
            **self.put_request(
                "/client/cancel_job", query.model_dump(exclude_none=True), **credentials
            )
        )
        return response.succeeded

    def fetch_jobs(
        self,
        job_ids: Sequence[str] | Sequence[uuid.UUID],
        **kwargs: object,
    ) -> dict[str, dict[str, str]]:
        query = _models.JobQuery(job_id=job_ids)
        credentials = self._extract_credentials({**kwargs, **self.client_kwargs})
        response = self.get_request(
            "/client/job", query.model_dump(exclude_none=True), **credentials
        )
        return {job_id: _models.JobData(**data).model_dump() for (job_id, data) in response.items()}

    def get_balance(self) -> dict[str, float]:
        response = _models.BalanceResponse(**self.get_request("/client/balance"))
        return {"balance": response.balance}

    def get_user_info(
        self,
        name: str | None = None,
        email: str | None = None,
        user_id: int | uuid.UUID | None = None,
    ) -> list[dict[str, str | float]]:
        if isinstance(user_id, int):
            raise TypeError("Superstaq API v0.3.0 uses UUID indexing for users, not integer.")
        query = _models.UserQuery(
            name=[name] if name is not None else None,
            email=[email] if email is not None else None,
            user_id=[user_id] if user_id is not None else None,
        )
        response = self.get_request("/client/user", query.model_dump(exclude_none=True))
        if not response:
            raise gss.SuperstaqServerException(
                "Something went wrong. The server has returned an empty response."
            )
        user_data = [_models.UserInfo(**data) for data in response]
        return [data.model_dump() for data in user_data]

    def _accept_terms_of_use(self, user_input: str) -> str:
        return self.put_request("/client/accept_terms_of_use", {"accept": user_input == "YES"})

    def get_targets(self, **kwargs: bool | None) -> list[gss.Target]:
        credentials = self._extract_credentials({**kwargs, **self.client_kwargs})
        query = _models.GetTargetsFilterModel(
            **{key: val for key, val in kwargs.items() if val is not None}
        )
        response = self.get_request(
            "/client/targets", query.model_dump(exclude_defaults=True), **credentials
        )
        targets = [_models.TargetModel(**data) for data in response]
        return [
            gss.typing.Target(
                target=target.target_name, **target.model_dump(exclude={"target_name"})
            )
            for target in targets
        ]

    def get_my_targets(self) -> list[gss.Target]:
        credentials = self._extract_credentials(self.client_kwargs)
        response = self.get_request("/client/targets", {"accessible": True}, **credentials)
        targets = [_models.TargetModel(**data) for data in response]
        return [
            gss.typing.Target(
                target=target.target_name, **target.model_dump(exclude={"target_name"})
            )
            for target in targets
        ]

    def target_info(self, target: str, **kwargs: object) -> dict[str, Any]:
        super().target_info(target)
        credentials = self._extract_credentials({**kwargs, **self.client_kwargs})

        response = _models.TargetInfo(
            **self.post_request(
                "/client/retrieve_target_info",
                _models.RetrieveTargetInfoModel(target=target, options_dict=kwargs).model_dump(),
                **credentials,
            )
        )
        return response.model_dump()

    def add_new_user(self, json_dict: dict[str, str]) -> str:
        new_user = _models.NewUser(**json_dict)
        return self.post_request("/client/user", new_user.model_dump(exclude_none=True))

    def update_user_balance(self, json_dict: dict[str, float | str]) -> str:
        user_email = json_dict.get("email")
        new_balance = json_dict.get("balance")
        if user_email is None:
            raise ValueError("Must provide a user email to update the balance of.")
        if new_balance is None:
            raise ValueError("Must provide a new balance to update the user with.")
        request = _models.UpdateUserDetails(balance=json_dict.get("balance"))
        return self.put_request(f"/client/user/{user_email}", request.model_dump(exclude_none=True))

    def update_user_role(self, json_dict: dict[str, int | str]) -> str:
        user_email = json_dict.get("email")
        new_role = json_dict.get("role")
        if user_email is None:
            raise ValueError("Must provide a user email to update the role of.")
        if new_role is None:
            raise ValueError("Must provide a new role to update the user with.")
        request = _models.UpdateUserDetails(role=json_dict.get("role"))
        return self.put_request(f"/client/user/{user_email}", request.model_dump(exclude_none=True))

    def resource_estimate(self, json_dict: dict[str, str]) -> dict[str, list[dict[str, int]]]:  # type: ignore [return]
        self._raise_not_implemented("resource_estimate")

    def aqt_compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        raise DeprecationWarning(
            "`aqt_compile` is deprecated for version 0.3.0. Use `compile` instead."
        )

    def qscout_compile(self, json_dict: dict[str, str]) -> dict[str, str | list[str]]:
        raise DeprecationWarning(
            "`qscout_compile` is deprecated for version 0.3.0. Use `compile` instead."
        )

    def compile(self, json_dict: dict[str, str]) -> dict[str, str]:
        circuits, circuit_type = self._extract_circuits(json_dict)

        # Define job
        new_job = _models.NewJob(
            job_type=_models.JobType.COMPILE,
            target=json_dict["target"],
            circuits=circuits,
            circuit_type=circuit_type,
            options_dict=json.loads(json_dict.get("options", "{}")),
        )
        # Submit job and store ID
        response = _models.NewJobResponse(**self.post_request("/client/job", new_job.model_dump()))
        job_id = str(response.job_id)
        job_data = self.fetch_jobs([job_id])[job_id]

        # Poll the server until all circuits have reached a terminal state.
        time_waited_seconds: float = 0.0
        while any(s not in _models.TERMINAL_CIRCUIT_STATES for s in job_data["statuses"]):
            # Status does a refresh.
            if time_waited_seconds > 7200:  # pragma: no cover
                raise TimeoutError(
                    f"Timed out while waiting for circuits to compile. The job ID is {job_id}. "
                    "Please use retrieve the job later when it has finished compiling."
                )
            time.sleep(2.5)
            time_waited_seconds += 2.5
            job_data = self.fetch_jobs([job_id])[job_id]

        # Exception if any have not been successful
        if not all(s == _models.CircuitStatus.COMPLETED for s in job_data["statuses"]):
            raise gss.SuperstaqException(
                f"Not all circuits were successfully compiled. Check job ID {job_id} for further "
                "details."
            )

        # Join circuits together in json string - TODO: make this neater.
        # Note mypy does not recognize that the above checks ensure there are no None's anywhere,
        # hence the ignored [arg-type]'s
        compile_dict = {
            "initial_logical_to_physicals": "["
            + ", ".join(str(dico) for dico in job_data["initial_logical_to_physicals"])
            + "]",
            "final_logical_to_physicals": "["
            + ", ".join(str(dico) for dico in job_data["final_logical_to_physicals"])
            + "]",
        }

        if circuit_type == _models.CircuitType.CIRQ:
            compile_dict["cirq_circuits"] = "[" + ", ".join(job_data["compiled_circuits"]) + "]"
            return compile_dict

        if circuit_type == _models.CircuitType.QISKIT:
            compile_dict["qiskit_circuits"] = "[" + ", ".join(job_data["compiled_circuits"]) + "]"
            return compile_dict

    def submit_qubo(  # type: ignore [return]
        self,
        qubo: Mapping[tuple[TQuboKey, ...], float],
        target: str,
        repetitions: int,
        method: str = "sim_anneal",
        max_solutions: int | None = 1000,
        *,
        qaoa_depth: int = 1,
        rqaoa_cutoff: int = 0,
        dry_run: bool = False,
        random_seed: int | None = None,
        **kwargs: object,
    ) -> dict[str, str]:
        self._raise_not_implemented("submit_qubo")

    def supercheq(
        self,
        files: list[list[int]],
        num_qubits: int,
        depth: int,
        circuit_return_type: str,
    ) -> Any:
        self._raise_not_implemented("supercheq")

    def submit_dfe(  # type: ignore [return]
        self,
        circuit_1: dict[str, str],
        target_1: str,
        circuit_2: dict[str, str],
        target_2: str,
        num_random_bases: int,
        shots: int,
        **kwargs: Any,
    ) -> list[str]:
        self._raise_not_implemented("submit_dfe")

    def process_dfe(self, job_ids: Sequence[str] | Sequence[uuid.UUID]) -> float:  # type: ignore [return]
        self._raise_not_implemented("process_dfe")

    def submit_aces(  # type: ignore [return]
        self,
        target: str,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: str | None = None,
        noise: dict[str, object] | None = None,
        tag: str | None = None,
        lifespan: int | None = None,
        weights: Sequence[int] | None = None,
    ) -> str:
        self._raise_not_implemented("submit_aces")

    def process_aces(self, job_id: str | uuid.UUID) -> list[float]:  # type: ignore [return]
        self._raise_not_implemented("process_aces")

    def submit_cb(  # type: ignore [return]
        self,
        target: str,
        shots: int,
        serialized_circuits: dict[str, str],
        n_channels: int,
        n_sequences: int,
        depths: Sequence[int],
        method: str | None = None,
        noise: dict[str, object] | None = None,
    ) -> str:
        self._raise_not_implemented("submit_cb")

    def process_cb(  # type: ignore [return]
        self, job_id: str | uuid.UUID, counts: str | None = None
    ) -> dict[str, Any]:
        self._raise_not_implemented("process_cb")

    def aqt_upload_configs(self, aqt_configs: dict[str, str]) -> str:
        response = self.put_request("/aqt_configs", _models.AQTConfigs(**aqt_configs).model_dump())
        return response

    def aqt_get_configs(self) -> dict[str, str]:
        response = _models.AQTConfigs(**self.get_request("/aqt_configs"))
        return response.model_dump()


def read_ibm_credentials(ibmq_name: str | None) -> dict[str, str]:
    """Function to try to read IBM credentials from .qiskit/qiskit-ibm.json.

    Args:
        ibmq_name: The name under which the IBM account credentials are locally stored.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If the provided `ibmq_name` does not have credentials stored in the
            config file or `token` and/or `channel` keys are missing for the credentials
            of the account under `ibmq_name`.
        ValueError: If no `ibmq_name` is provided and multiple accounts are found with
            none marked as default.

    Returns:
        Dictionary containing the ibm token, channel, and instance (if available).
    """
    config_dir = pathlib.Path.home().joinpath(".qiskit")
    path = config_dir.joinpath("qiskit-ibm.json")
    if path.is_file():
        with open(path) as config_file:
            config = json.load(config_file)
        if ibmq_name is None:
            if len(config) == 1:
                ibmq_name = next(iter(config.keys()))
            elif any(creds.get("is_default_account") for creds in config.values()):
                ibmq_name = next(name for name in config if config[name].get("is_default_account"))
            else:
                raise ValueError(
                    "Multiple accounts found but none are marked as default.",
                    " Please provide the name of the account to retrieve.",
                )
        elif ibmq_name not in config:
            raise KeyError(
                f"No account credentials saved under the name '{ibmq_name}'"
                f" in the config file found at '{path}'."
            )

        credentials = config.get(ibmq_name)
        if any(key not in credentials for key in ["token", "channel"]):
            raise KeyError(
                "`token` and/or `channel` keys missing from credentials for the account",
                f" under the name '{ibmq_name}' in the file '{path}'.",
            )

        return credentials

    raise FileNotFoundError(f"The `qiskit-ibm.json` file was not found in '{config_dir}'.")


def find_api_key() -> str:
    """Function to try to load a Superstaq API key from the environment or a key file.

    Raises:
        OSError: If the Superstaq API key could not be found in the environment.
        EnvironmentError: If the Superstaq API key could not be found.

    Returns:
        Superstaq API key string.
    """
    # look for the key in the environment
    env_api_key = os.getenv("SUPERSTAQ_API_KEY")
    if env_api_key:
        return env_api_key

    data_dir = pathlib.Path(os.getenv("XDG_DATA_HOME", "~/.local/share")).expanduser()
    home_dir = pathlib.Path.home()
    for directory in [
        data_dir.joinpath("super.tech"),
        data_dir.joinpath("coldquanta"),
        home_dir.joinpath(".super.tech"),
        home_dir.joinpath(".coldquanta"),
    ]:
        path = directory.joinpath("superstaq_api_key")
        if path.is_file():
            with open(path) as file:
                return file.readline()

    raise OSError(
        "Superstaq API key not specified and not found.\n"
        "Try passing an 'api_key' variable, or setting your API key in the command line "
        "with SUPERSTAQ_API_KEY=...\n"
        "Please visit https://superstaq.readthedocs.io/en/latest/get_started/credentials.html to "
        "access your API key."
    )
