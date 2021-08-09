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
"""Client for making requests to SuperstaQ's API."""

import json
import sys
import time
import urllib
from typing import Any, Callable, cast, Dict, Optional

import requests

import cirq_superstaq


class _SuperstaQClient:
    """Handles calls to SuperstaQ's API.

    Users should not instantiate this themselves, but instead should use `cirq_superstaq.Service`.
    """

    RETRIABLE_STATUS_CODES = {
        requests.codes.internal_server_error,
        requests.codes.service_unavailable,
    }
    SUPPORTED_TARGETS = {"qpu", "simulator"}
    SUPPORTED_VERSIONS = {
        "v0.1",
    }

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        default_target: Optional[str] = None,
        api_version: str = cirq_superstaq.API_VERSION,
        max_retry_seconds: float = 3600,  # 1 hour
        verbose: bool = False,
        ibmq_token: str = None,
        ibmq_group: str = None,
        ibmq_project: str = None,
        ibmq_hub: str = None,
        ibmq_pulse: bool = True,
    ):
        """Creates the SuperstaQClient.

        Users should use `cirq_superstaq.Service` instead of this class directly.

        The SuperstaQClient handles making requests to the SuperstaQClient,
        returning dictionary results. It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the SuperstaQ API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_key: The key used for authenticating against the SuperstaQ API.
            default_target: The default target to run against. Supports one of 'qpu' and
                'simulator'. Can be overridden by calls with target in their signature.
            api_version: Which version fo the api to use. As of Dec, 2020, accepts 'v0.1' only,
                which is the default.
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
        """
        url = urllib.parse.urlparse(remote_host)
        assert url.scheme and url.netloc, (
            f"Specified remote_host {remote_host} is not a valid url, for example "
            "http://example.com"
        )
        assert (
            api_version in self.SUPPORTED_VERSIONS
        ), f"Only api v0.1 is accepted but was {api_version}"
        assert (
            default_target is None or default_target in self.SUPPORTED_TARGETS
        ), f"Target can only be one of {self.SUPPORTED_TARGETS} but was {default_target}."
        assert max_retry_seconds >= 0, "Negative retry not possible without time machine."

        self.url = f"{url.scheme}://{url.netloc}/{api_version}"
        self.headers = {"Authorization": api_key, "Content-Type": "application/json"}
        self.default_target = default_target
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        self.ibmq_token = ibmq_token
        self.ibmq_group = ibmq_group
        self.ibmq_project = ibmq_project
        self.ibmq_hub = ibmq_hub
        self.ibmq_pulse = ibmq_pulse

    def create_job(
        self,
        serialized_program: str,
        repetitions: Optional[int] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> dict:
        """Create a job.

        Args:
            serialized_program: The `cirq_superstaq.SerializedProgram` containing the serialized
                information about the circuit to run.
            repetitions: The number of times to repeat the circuit. For simulation the repeated
                sampling is not done on the server, but is passed as metadata to be recovered
                from the returned job.
            target: If supplied the target to run on. Supports one of `qpu` or `simulator`. If not
                set, uses `default_target`.
            name: An optional name of the job. Different than the `job_id` of the job.

        Returns:
            The json body of the response as a dict. This does not contain populated information
            about the job, but does contain the job id.

        Raises:
            An SuperstaQException if the request fails.
        """
        actual_target = self._target(target)
        json_dict: Dict[str, Any] = {
            "circuit": json.loads(serialized_program),
            "backend": actual_target,
            "shots": repetitions,
            "ibmq_token": self.ibmq_token,
            "ibmq_group": self.ibmq_group,
            "ibmq_project": self.ibmq_project,
            "ibmq_hub": self.ibmq_hub,
            "ibmq_pulse": self.ibmq_pulse,
        }

        def request() -> requests.Response:
            return requests.post(
                f"{self.url}/job",
                json=json_dict,
                headers=self.headers,
                verify=(cirq_superstaq.API_URL == self.url),
            )

        return self._make_request(request).json()

    def get_job(self, job_id: str) -> dict:
        """Get the job from the SuperstaQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Returns:
            The json body of the response as a dict.

        Raises:
            SuperstaQNotFoundException: If a job with the given job_id does not exist.
            SuperstaQException: For other API call failures.
        """

        def request() -> requests.Response:
            return requests.get(
                f"{self.url}/job/{job_id}",
                headers=self.headers,
                verify=(cirq_superstaq.API_URL == self.url),
            )

        return self._make_request(request).json()

    def aqt_compile(self, serialized_program: str) -> dict:
        """Makes a POST request to SuperstaQ API to compile a circuit for Berkeley-AQT."""
        json_dict = {"circuit": json.loads(serialized_program)}

        def request() -> requests.Response:
            return requests.post(
                f"{self.url}/aqt_compile",
                headers=self.headers,
                json=json_dict,
                verify=(cirq_superstaq.API_URL == self.url),
            )

        return self._make_request(request).json()

    def _target(self, target: Optional[str]) -> str:
        """Returns the target if not None or the default target.

        Raises:
            AssertionError: if both `target` and `default_target` are not set.
        """
        assert target is not None or self.default_target is not None, (
            "One must specify a target on this call, or a default_target on the service/client, "
            "but neither were set."
        )
        return cast(str, target or self.default_target)

    def _make_request(self, request: Callable[[], requests.Response]) -> requests.Response:
        """Make a request to the API, retrying if necessary.

        Args:
            request: A function that returns a `requests.Response`.

        Raises:
            SuperstaQException: If there was a not-retriable error from the API.
            TimeoutError: If the requests retried for more than `max_retry_seconds`.

        Returns:
            The request.Response from the final successful request call.
        """
        # Initial backoff of 100ms.
        delay_seconds = 0.1
        while True:
            try:
                response = request()
                if response.ok:
                    return response
                if response.status_code == requests.codes.unauthorized:
                    raise cirq_superstaq.superstaq_exceptions.SuperstaQException(
                        '"Not authorized" returned by SuperstaQ API.  '
                        "Check to ensure you have supplied the correct API key.",
                        response.status_code,
                    )
                if response.status_code == requests.codes.not_found:
                    raise cirq_superstaq.superstaq_exceptions.SuperstaQNotFoundException(
                        "SuperstaQ could not find requested resource."
                    )
                if response.status_code not in self.RETRIABLE_STATUS_CODES:
                    raise cirq_superstaq.superstaq_exceptions.SuperstaQException(
                        "Non-retry-able error making request to SuperstaQ API. "
                        f"Status: {response.status_code} "
                        f"Error :{response.reason}",
                        response.status_code,
                    )
                message = response.reason
                # Fallthrough should retry.
            except requests.RequestException as e:
                # Connection error, timeout at server, or too many redirects.
                # Retry these.
                message = f"RequestException of type {type(e)}."
            if delay_seconds > self.max_retry_seconds:
                raise TimeoutError(f"Reached maximum number of retries. Last error: {message}")
            if self.verbose:
                print(message, file=sys.stderr)
                print(f"Waiting {delay_seconds} seconds before retrying.")
            time.sleep(delay_seconds)
            delay_seconds *= 2
