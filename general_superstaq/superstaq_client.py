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
import textwrap
import time
import urllib
from typing import Any, Callable, Dict, List, Optional, Union

import qubovert as qv
import requests

import general_superstaq as gss
from general_superstaq.typing import MaxSharpeJson, MinVolJson, TSPJson, WareHouseJson


class _SuperstaQClient:
    """Handles calls to SuperstaQ's API.

    Users should not instantiate this themselves,
    but instead should use `$client_superstaq.Service`.
    """

    RETRIABLE_STATUS_CODES = {
        requests.codes.service_unavailable,
    }
    SUPPORTED_VERSIONS = {
        gss.API_VERSION,
    }

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        client_name: str,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
    ):
        """Creates the SuperstaQClient.

        Users should use `$client_superstaq.Service` instead of this class directly.

        The SuperstaQClient handles making requests to the SuperstaQClient,
        returning dictionary results. It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the SuperstaQ API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_key: The key used for authenticating against the SuperstaQ API.
            api_version: Which version fo the api to use, defaults to client_superstaq.API_VERSION,
                which is the most recent version when this client was downloaded.
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
        """

        self.api_key = api_key
        self.client_name = client_name
        self.api_version = api_version
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        url = urllib.parse.urlparse(remote_host)
        assert url.scheme and url.netloc, (
            f"Specified remote_host {remote_host} is not a valid url, for example "
            "http://example.com"
        )

        assert (
            self.api_version in self.SUPPORTED_VERSIONS
        ), f"Only API versions {self.SUPPORTED_VERSIONS} are accepted but got {self.api_version}"
        assert max_retry_seconds >= 0, "Negative retry not possible without time machine."

        self.url = f"{url.scheme}://{url.netloc}/{api_version}"
        self.verify_https: bool = f"{gss.API_URL}/{self.api_version}" == self.url
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "X-Client-Name": self.client_name,
            "X-Client-Version": self.api_version,
        }

    def get_request(self, endpoint: str) -> Any:
        def request() -> requests.Response:
            return requests.get(
                f"{self.url}{endpoint}",
                headers=self.headers,
                verify=self.verify_https,
            )

        return self._make_request(request).json()

    def post_request(self, endpoint: str, json_dict: Dict[str, Any]) -> Any:
        def request() -> requests.Response:
            return requests.post(
                f"{self.url}{endpoint}",
                json=json_dict,
                headers=self.headers,
                verify=self.verify_https,
            )

        return self._make_request(request).json()

    def create_job(
        self,
        serialized_circuits: Dict[str, str],
        repetitions: int = 1,
        target: str = "ss_unconstrained_simulator",
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """Create a job.

        Args:
            serialized_circuits: The serialized representation of the circuit to run.
            repetitions: The number of times to repeat the circuit. For simulation the repeated
                sampling is not done on the server, but is passed as metadata to be recovered
                from the returned job.
            target: Target to run on.
            ibmq_pulse: Specify whether to run the job using SuperstaQ's pulse-level optimizations

        Returns:
            The json body of the response as a dict. This does not contain populated information
            about the job, but does contain the job id.

        Raises:
            An SuperstaQException if the request fails.
        """
        json_dict: Dict[str, Any] = {
            **serialized_circuits,
            "target": target,
            "shots": repetitions,
        }

        if method is not None:
            json_dict["method"] = method

        if options is not None:
            json_dict["options"] = json.dumps(options)
        return self.post_request("/jobs", json_dict)

    def get_job(self, job_id: str) -> Dict[str, str]:
        """Get the job from the SuperstaQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Returns:
            The json body of the response as a dict.

        Raises:
            SuperstaQNotFoundException: If a job with the given job_id does not exist.
            SuperstaQException: For other API call failures.
        """
        return self.get_request(f"/job/{job_id}")

    def get_balance(self) -> Dict[str, float]:
        """Get the querying user's account balance in USD.

        Returns:
            The json body of the response as a dict.
        """
        return self.get_request("/balance")

    def get_targets(self) -> Dict[str, Dict[str, List[str]]]:
        """Makes a GET request to SuperstaQ API to get a list of available targets."""
        return self.get_request("/targets")

    def ibmq_set_token(self, ibmq_token: Dict[str, str]) -> str:
        """Makes a POST request to SuperstaQ API to set IBMQ token field in database.

        Args:
            ibmq_token: dictionary with IBMQ token string entry.

        Returns:
            The json body of the response as a dict.
        """

        def request() -> requests.Response:
            return requests.post(
                f"{self.url}/ibmq_token",
                headers=self.headers,
                json=ibmq_token,
                verify=self.verify_https,
            )

        return self._make_request(request).json()

    def resource_estimate(self, json_dict: Dict[str, str]) -> Dict[str, List[Dict[str, int]]]:
        return self.post_request("/resource_estimate", json_dict)

    def aqt_compile(self, json_dict: Dict[str, Union[int, str, List[str]]]) -> Dict[str, str]:
        """Makes a POST request to SuperstaQ API to compile a list of circuits for Berkeley-AQT."""
        return self.post_request("/aqt_compile", json_dict)

    def qscout_compile(
        self, json_dict: Dict[str, Union[str, List[str]]]
    ) -> Dict[str, Union[str, List[str]]]:
        """Makes a POST request to SuperstaQ API to compile a list of circuits for QSCOUT."""
        return self.post_request("/qscout_compile", json_dict)

    def cq_compile(self, json_dict: Dict[str, Union[str, List[str]]]) -> Dict[str, str]:
        """Makes a POST request to SuperstaQ API to compile a list of circuits for CQ."""
        return self.post_request("/cq_compile", json_dict)

    def ibmq_compile(self, json_dict: Dict[str, Union[str, List[str]]]) -> Dict[str, str]:
        """Makes a POST request to SuperstaQ API to compile a circuits for IBM devices."""
        return self.post_request("/ibmq_compile", json_dict)

    def neutral_atom_compile(self, json_dict: Dict[str, Union[str, List[str]]]) -> Dict[str, str]:
        """Makes a POST request to SuperstaQ API to compile a circuits for neutral atom devices."""
        return self.post_request("/neutral_atom_compile", json_dict)

    def submit_qubo(self, qubo: qv.QUBO, target: str, repetitions: int = 1000) -> Dict[str, str]:
        """Makes a POST request to SuperstaQ API to submit a QUBO problem to the given target."""
        json_dict = {
            "qubo": gss.qubo.convert_qubo_to_model(qubo),
            "target": target,
            "shots": repetitions,
        }
        return self.post_request("/qubo", json_dict)

    def find_min_vol_portfolio(
        self, json_dict: Dict[str, Union[List[str], int, float, str]]
    ) -> MinVolJson:
        """Makes a POST request to SuperstaQ API to find a minimum volatility portfolio
        that exceeds a certain specified return."""
        return self.post_request("/minvol", json_dict)

    def find_max_pseudo_sharpe_ratio(
        self, json_dict: Dict[str, Union[List[str], float, str, Optional[int]]]
    ) -> MaxSharpeJson:
        """Makes a POST request to SuperstaQ API to find a max Sharpe ratio portfolio."""
        return self.post_request("/maxsharpe", json_dict)

    def tsp(self, json_dict: Dict[str, List[str]]) -> TSPJson:
        """Makes a POST request to SuperstaQ API to find a optimal TSP tour."""
        return self.post_request("/tsp", json_dict)

    def warehouse(self, json_dict: Dict[str, Union[int, List[str], str]]) -> WareHouseJson:
        """Makes a POST request to SuperstaQ API to find optimal warehouse assignment."""
        return self.post_request("/warehouse", json_dict)

    def aqt_upload_configs(self, aqt_configs: Dict[str, str]) -> str:
        """Makes a POST request to SuperstaQ API to upload configurations."""
        return self.post_request("/aqt_configs", aqt_configs)

    def aqt_get_configs(self) -> Dict[str, str]:
        """Writes AQT configs from the AQT system onto the given file paths."""
        return self.get_request("/get_aqt_configs")

    def _handle_status_codes(self, response: requests.Response) -> None:
        if response.status_code == requests.codes.unauthorized:
            raise gss.SuperstaQException(
                '"Not authorized" returned by SuperstaQ API.  '
                "Check to ensure you have supplied the correct API key.",
                response.status_code,
            )

        if response.status_code not in self.RETRIABLE_STATUS_CODES:
            if "message" in response.json():
                message = response.json()["message"]
            else:
                message = str(response.text)
            raise gss.SuperstaQException(
                f"Non-retriable error making request to SuperstaQ API, {message}",
                response.status_code,
            )

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
                print(message, file=sys.stderr)
                print(f"Waiting {delay_seconds} seconds before retrying.")
            time.sleep(delay_seconds)
            delay_seconds *= 2

    def __str__(self) -> str:
        return f"Client with host={self.url} and name={self.client_name}"

    def __repr__(self) -> str:
        return textwrap.dedent(
            f"""\
            gss.superstaq_client._SuperstaQClient(
                remote_host={self.url!r},
                api_key={self.api_key!r},
                client_name={self.client_name!r},
                api_version={self.api_version!r},
                max_retry_seconds={self.max_retry_seconds!r},
                verbose={self.verbose!r},
            )"""
        )
