"""Client for making requests to SuperstaQ's API."""

import urllib
from typing import Optional

import requests

import applications_superstaq


class _SuperstaQClient:
    """Handles calls to SuperstaQ's API.

    Users should not instantiate this themselves, but instead should use
    `applications_superstaq.Service`.
    """

    RETRIABLE_STATUS_CODES = {
        requests.codes.service_unavailable,
    }
    SUPPORTED_TARGETS = {"qpu", "simulator"}
    SUPPORTED_VERSIONS = {
        applications_superstaq.API_VERSION,
    }

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        default_target: Optional[str] = None,
        api_version: str = applications_superstaq.API_VERSION,
        max_retry_seconds: float = 3600,  # 1 hour
        verbose: bool = False,
        ibmq_token: str = None,
        ibmq_group: str = None,
        ibmq_project: str = None,
        ibmq_hub: str = None,
        ibmq_pulse: bool = True,
    ):
        """Creates the SuperstaQClient.

        Users should use `applications_superstaq.Service` instead of this class directly.

        The SuperstaQClient handles making requests to the SuperstaQClient,
        returning dictionary results. It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the SuperstaQ API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_key: The key used for authenticating against the SuperstaQ API.
            default_target: The default target to run against. Supports one of 'qpu' and
                'simulator'. Can be overridden by calls with target in their signature.
            api_version: Which version fo the api to use, defaults to
                applications_superstaq.API_VERSION, which is the most recent version when this
                client was downloaded.
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
        ), f"Only api versions {self.SUPPORTED_VERSIONS} are accepted but was {api_version}"
        assert (
            default_target is None or default_target in self.SUPPORTED_TARGETS
        ), f"Target can only be one of {self.SUPPORTED_TARGETS} but was {default_target}."
        assert max_retry_seconds >= 0, "Negative retry not possible without time machine."

        self.url = f"{url.scheme}://{url.netloc}/{api_version}"
        self.verify_https: bool = (
            applications_superstaq.API_URL + "/" + applications_superstaq.API_VERSION == self.url
        )
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
            "X-Client-Name": "applications-superstaq",
            "X-Client-Version": applications_superstaq.API_VERSION,
        }
        self.default_target = default_target
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        self.ibmq_token = ibmq_token
        self.ibmq_group = ibmq_group
        self.ibmq_project = ibmq_project
        self.ibmq_hub = ibmq_hub
        self.ibmq_pulse = ibmq_pulse

    def _handle_status_codes(self, response: requests.Response) -> None:
        if response.status_code == requests.codes.unauthorized:
            raise applications_superstaq.superstaq_exceptions.SuperstaQException(
                '"Not authorized" returned by SuperstaQ API.  '
                "Check to ensure you have supplied the correct API key.",
                response.status_code,
            )
        if response.status_code == requests.codes.not_found:
            raise applications_superstaq.superstaq_exceptions.SuperstaQNotFoundException(
                "SuperstaQ could not find requested resource."
            )

        if response.status_code not in self.RETRIABLE_STATUS_CODES:
            message = response.reason
            if response.status_code == 400:
                message = str(response.text)
            raise applications_superstaq.superstaq_exceptions.SuperstaQException(
                "Non-retriable error making request to SuperstaQ API. "
                f"Status: {response.status_code} "
                f"Error : {message}",
                response.status_code,
            )
