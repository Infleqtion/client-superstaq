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
import json
import os
import pathlib
import sys
import textwrap
import time
import urllib
from typing import Any, Callable, Dict, List, Optional, Union

import qubovert as qv
import requests

import general_superstaq as gss


class _SuperstaqClient:
    """Handles calls to Superstaq's API.

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
        client_name: str,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
    ):
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
            api_version: Which version fo the api to use, defaults to client_superstaq.API_VERSION,
                which is the most recent version when this client was downloaded.
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
        """

        self.api_key = api_key or gss.superstaq_client.find_api_key()
        self.remote_host = remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or gss.API_URL
        self.client_name = client_name
        self.api_version = api_version
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        url = urllib.parse.urlparse(self.remote_host)
        assert url.scheme and url.netloc, (
            f"Specified remote_host {self.remote_host} is not a valid url, for example "
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
        """Performs a GET request on a given endpoint.

        Args:
            endpoint: The endpoint to perform the GET request on.

        Returns:
            The response of the GET request.
        """

        def request() -> requests.Response:
            """Builds GET request object.

            Returns:
                The Flask GET request object.
            """
            return requests.get(
                f"{self.url}{endpoint}",
                headers=self.headers,
                verify=self.verify_https,
            )

        return self._make_request(request).json()

    def get_superstaq_version(self) -> Dict[str, Optional[str]]:
        """Gets Superstaq version from response header.

        Returns:
            A dict containing the current Superstaq version.
        """

        response = requests.get(self.url)
        version = response.headers.get("superstaq_version")

        return {"superstaq_version": version}

    def post_request(self, endpoint: str, json_dict: Dict[str, Any]) -> Any:
        """Performs a POST request on a given endpoint with a given payload.

        Args:
            endpoint: The endpoint to perform the POST request on.
            json_dict: The payload to POST.

        Returns:
            The response of the POST request.
        """

        def request() -> requests.Response:
            """Builds GET request object.

            Returns:
                The Flask GET request object.
            """
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
        **kwargs: Any,
    ) -> Dict[str, List[str]]:
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
                - qiskit_pulse: Whether to use Superstaq's pulse-level optimizations for IBMQ
                devices.

        Returns:
            The json body of the response as a dict. This does not contain populated information
            about the job, but does contain the job id.

        Raises:
            An SuperstaqException if the request fails.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_integer_param(repetitions)

        json_dict: Dict[str, Any] = {
            **serialized_circuits,
            "target": target,
            "shots": int(repetitions),
        }

        if method is not None:
            json_dict["method"] = method

        if kwargs:
            json_dict["options"] = json.dumps(kwargs)
        return self.post_request("/jobs", json_dict)

    def get_job(self, job_id: str) -> Dict[str, str]:
        """Get the job from the Superstaq API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Returns:
            The json body of the response as a dict.

        Raises:
            SuperstaqNotFoundException: If a job with the given `job_id` does not exist.
            SuperstaqException: For other API call failures.
        """
        return self.get_request(f"/job/{job_id}")

    def get_balance(self) -> Dict[str, float]:
        """Get the querying user's account balance in USD.

        Returns:
            The json body of the response as a dict.
        """
        return self.get_request("/balance")

    def _accept_terms_of_use(self, user_input: str) -> str:
        """Makes a POST request to Superstaq API to confirm acceptance of terms of use.

        Args:
            user_input: The user's response to prompt for acceptance of TOU. Server accepts YES.

        Returns:
            String with success message.
        """
        return self.post_request("/accept_terms_of_use", {"user_input": user_input})

    def get_targets(self) -> Dict[str, Dict[str, List[str]]]:
        """Makes a GET request to retrieve targets from the Superstaq API.

        Gets a list of available, unavailable, and retired targets.

        Returns:
            A dictionary listing the targets.
        """
        return self.get_request("/targets")

    def add_new_user(self, json_dict: Dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to add a new user.

        Args:
            json_dict: The dictionary with user entry.

        Returns:
            The response as a string.
        """
        return self.post_request("/add_new_user", json_dict)

    def update_user_balance(self, json_dict: Dict[str, Union[float, str]]) -> str:
        """Makes a POST request to Superstaq API to update a user's balance in the database.

        Args:
            json_dict: The dictionary with user entry and new balance.

        Returns:
            The response as a string.
        """
        return self.post_request("/update_user_balance", json_dict)

    def update_user_role(self, json_dict: Dict[str, Union[int, str]]) -> str:
        """Makes a POST request to Superstaq API to update a user's role.

        Args:
            json_dict: The dictionary with user entry and new role.

        Returns:
            The response as a string.
        """
        return self.post_request("/update_user_role", json_dict)

    def resource_estimate(self, json_dict: Dict[str, str]) -> Dict[str, List[Dict[str, int]]]:
        """POSTs the given payload to the `/resource_estimate` endpoint.

        Args:
            json_dict: The payload to POST.

        Returns:
            The response of the given payload.
        """
        return self.post_request("/resource_estimate", json_dict)

    def aqt_compile(self, json_dict: Dict[str, str]) -> Dict[str, str]:
        """Makes a POST request to Superstaq API to compile a list of circuits for Berkeley-AQT.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit(s) data.
        """
        return self.post_request("/aqt_compile", json_dict)

    def qscout_compile(self, json_dict: Dict[str, str]) -> Dict[str, Union[str, List[str]]]:
        """Makes a POST request to Superstaq API to compile a list of circuits for QSCOUT.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit(s) data.
        """
        return self.post_request("/qscout_compile", json_dict)

    def compile(self, json_dict: Dict[str, str]) -> Dict[str, str]:
        """Makes a POST request to Superstaq API to compile a list of circuits.

        Args:
            json_dict: The dictionary containing data to compile.

        Returns:
            A dictionary containing compiled circuit data.
        """
        return self.post_request("/compile", json_dict)

    def submit_qubo(
        self,
        qubo: qv.QUBO,
        target: str,
        repetitions: int = 1000,
        method: Optional[str] = None,
        maxout: Optional[int] = 1000,
    ) -> Dict[str, str]:
        """Makes a POST request to Superstaq API to submit a QUBO problem to the
        given target.

        Args:
            qubo: A `qv.QUBO` object.
            target: The target to submit the qubo.
            repetitions: Number of repetitions to use. Defaults to 1000.
            method: An optional method parameter. Defaults to None.
            maxout: An optional maxout paramter. Defaults to 1000.

        Returns:
            A dictionary from the POST request.
        """
        gss.validation.validate_target(target)
        gss.validation.validate_integer_param(repetitions)

        json_dict = {
            "qubo": gss.qubo.convert_qubo_to_model(qubo),
            "target": target,
            "shots": int(repetitions),
            "method": method,
            "maxout": maxout,
        }
        return self.post_request("/qubo", json_dict)

    def supercheq(
        self,
        files: List[List[int]],
        num_qubits: int,
        depth: int,
        circuit_return_type: str,
    ) -> Any:
        """Performs a POST request on the `/supercheq` endpoint.

        Args:
            files: List of files specified as binary using ints.
                For example: [[1, 0, 1], [1, 1, 1]].
            num_qubits: Number of qubits to run Supercheq on.
            depth: The depth of the circuits to run Supercheq on.
            circuit_return_type: Supports only `cirq` and `qiskit` for now.

        Returns:
            The output of Supercheq.
        """
        gss.validation.validate_integer_param(num_qubits)
        gss.validation.validate_integer_param(depth)

        json_dict = {
            "files": files,
            "num_qubits": int(num_qubits),
            "depth": int(depth),
            "circuit_return_type": circuit_return_type,
        }
        return self.post_request("/supercheq", json_dict)

    def target_info(self, target: str) -> Dict[str, Any]:
        """Makes a POST request to the /target_info endpoint.

        Uses the Superstaq API to request information about `target`.

        Args:
            target: A string representing the device to get information about.

        Returns:
            The target information.
        """
        gss.validation.validate_target(target)

        json_dict = {
            "target": target,
        }
        return self.post_request("/target_info", json_dict)

    def ibmq_set_token(self, json_dict: Dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to set IBMQ token field in database.

        Args:
            json_dict: Dictionary with IBMQ token string entry.

        Returns:
            The response as a string.
        """
        return self.post_request("/ibmq_token", json_dict)

    def cq_set_token(self, json_dict: Dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to set CQ token field in database.

        Args:
            json_dict: Dictionary with CQ token string entry.

        Returns:
            The response as a string.
        """
        return self.post_request("/cq_token", json_dict)

    def aqt_upload_configs(self, aqt_configs: Dict[str, str]) -> str:
        """Makes a POST request to Superstaq API to upload configurations.

        Args:
            aqt_configs: The configs to be uploaded.

        Returns:
            A string response from POST request.
        """
        return self.post_request("/aqt_configs", aqt_configs)

    def aqt_get_configs(self) -> Dict[str, str]:
        """Writes AQT configs from the AQT system onto the given file path.

        Returns:
            A dictionary containing the AQT configs.
        """
        return self.get_request("/get_aqt_configs")

    def _handle_status_codes(self, response: requests.Response) -> None:
        """A method to handle status codes.

        Args:
            response: The `requests.Response` to get the status codes from.

        Raises:
            gss.SuperstaqException: If unauthorized by Superstaq API.
            gss.SuperstaqException: If an error has occurred in making a request
                to the Superstaq API.
        """
        if response.status_code == requests.codes.unauthorized:
            if response.json() == (
                "You must accept the Terms of Use (superstaq.super.tech/terms_of_use)."
            ):
                self._prompt_accept_terms_of_use()
                return
            else:
                raise gss.SuperstaqException(
                    '"Not authorized" returned by Superstaq API.  '
                    "Check to ensure you have supplied the correct API key.",
                    response.status_code,
                )

        if response.status_code not in self.RETRIABLE_STATUS_CODES:
            if "message" in response.json():
                message = response.json()["message"]
            else:
                message = str(response.text)
            slack_invite_url = (
                "https://join.slack.com/t/superstaq/shared_invite/"
                "zt-1wr6eok5j-fMwB7dPEWGG~5S474xGhxw"
            )
            raise gss.SuperstaqException(
                f"Non-retriable error making request to Superstaq API, {message}.\n\n"
                "If you would like to contact a member of our team, email us at "
                f"superstaq@infleqtion.com or join our Slack workspace: {slack_invite_url}",
                response.status_code,
            )

    def _prompt_accept_terms_of_use(self) -> None:
        """Prompts terms of use.

        Raises:
            gss.SuperstaqException: If terms of use are not accepted.
        """
        message = (
            "Acceptance of the Terms of Use (superstaq.super.tech/terms_of_use)"
            " is necessary before using Superstaq.\nType in YES to accept: "
        )
        user_input = input(message).upper()
        response = self._accept_terms_of_use(user_input)
        print(response)
        if response != "Accepted. You can now continue using Superstaq.":
            raise gss.SuperstaqException(
                "You'll need to accept Terms of Use before usage of Superstaq.",
                requests.codes.unauthorized,
            )

    def _make_request(self, request: Callable[[], requests.Response]) -> requests.Response:
        """Make a request to the API, retrying if necessary.

        Args:
            request: A function that returns a `requests.Response`.

        Raises:
            SuperstaqException: If there was a not-retriable error from the API.
            TimeoutError: If the requests retried for more than `max_retry_seconds`.

        Returns:
            The `request.Response` from the final successful request call.
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
            gss.superstaq_client._SuperstaqClient(
                remote_host={self.url!r},
                api_key={self.api_key!r},
                client_name={self.client_name!r},
                api_version={self.api_version!r},
                max_retry_seconds={self.max_retry_seconds!r},
                verbose={self.verbose!r},
            )"""
        )


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
            with open(path, "r") as file:
                return file.readline()

    raise EnvironmentError(
        "Superstaq API key not specified and not found.\n"
        "Try passing an 'api_key' variable, or setting your API key in the command line "
        "with SUPERSTAQ_API_KEY=..."
    )
