from __future__ import annotations

import general_superstaq as gss
import pytest

import qiskit_superstaq as qss


class MockSuperstaqClient(gss.superstaq_client._SuperstaqClient):
    """Stand-in for `_SuperstaqClient` that the tests can call."""

    def get_targets(self, **kwargs: bool | None) -> list[gss.typing.Target]:
        """Makes a GET request to retrieve targets from the Superstaq API.

        Args:
            kwargs: Optional flags to restrict/filter returned targets.
                - simulator: Optional flag to restrict the list of targets to (non-) simulators.
                - supports_submit: Optional boolean flag to only return targets that (don't) allow
                   circuit submissions.
                - supports_submit_qubo: Optional boolean flag to only return targets that (don't)
                   allow qubo submissions.
                - supports_compile: Optional boolean flag to return targets that (don't) support
                   circuit compilation.
                - available: Optional boolean flag to only return targets that are (not) available
                   to use.
                - retired: Optional boolean flag to only return targets that are or are not retired.

        Returns:
            A list of Superstaq targets matching all provided criteria.
        """
        return gss.testing.RETURNED_TARGETS

    def target_info(self, target: str) -> dict[str, object]:
        """Mocks a request to the /target_info endpoint.

        Args:
            target: A string representing the device to get information about.

        Returns:
            The target information.
        """
        return {
            "target_info": {
                "target": target,
                "num_qubits": 4,
                "basis_gates": None,
                "coupling_map": [[0, 1], [1, 2]],
            },
        }


class MockSuperstaqProvider(qss.SuperstaqProvider):
    """Stand-in for `SuperstaqProvider` that the tests can call."""

    def __init__(
        self,
        api_key: str | None = None,
        remote_host: str | None = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        """Initializes a `SuperstaqProvider`.

        Args:
            api_key: A string that allows access to the Superstaq API. If no key is provided, then
                this instance tries to use the environment variable `SUPERSTAQ_API_KEY`. If
                `SUPERSTAQ_API_KEY` is not set, then this instance checks for the
                following files:
                - `$XDG_DATA_HOME/super.tech/superstaq_api_key`
                - `$XDG_DATA_HOME/coldquanta/superstaq_api_key`
                - `~/.super.tech/superstaq_api_key`
                - `~/.coldquanta/superstaq_api_key`
                If one of those files exists, then it is treated as a plain text file, and the first
                line of this file is interpreted as an API key.  Failure to find an API key raises
                an `EnvironmentError`.
            remote_host: The location of the API in the form of a URL. If this is None,
                then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
                If that variable is not set, then this uses
                `https://superstaq.infleqtion.com/{api_version}`,
                where `{api_version}` is the `api_version` specified below.
            api_version: The version of the API.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            EnvironmentError: If an API key was not provided and could not be found.
        """
        self._name = "mock_superstaq_provider"

        self._client = MockSuperstaqClient(
            client_name="qiskit-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )


@pytest.fixture()
def fake_superstaq_provider() -> MockSuperstaqProvider:
    """Fixture that retrieves the `SuperstaqProvider`.

    Returns:
        The Mock Superstaq provider.
    """
    return MockSuperstaqProvider(api_key="MY_TOKEN")
