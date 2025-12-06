# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import random

import general_superstaq as gss
import numpy as np
import pytest
from general_superstaq.testing import RETURNED_TARGETS

import qiskit_superstaq as qss

# mypy: disable-error-code="empty-body"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(random.getrandbits(128))


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
        return RETURNED_TARGETS

    def target_info(self, target: str, **kwargs: object) -> dict[str, object]:
        """Mocks a request to the /target_info endpoint.

        Args:
            target: A string representing the device to get information about.
            kwargs: Any other information.

        Returns:
            The target information.
        """
        return {
            "target_info": {
                "target": target,
                "num_qubits": 4,
                "native_gate_set": ["cz", "id", "rz", "sx", "x", "gr"],
                "coupling_map": [[0, 1], [1, 2]],
                "acquire_alignment": 1,
                "granularity": 1,
                "min_length": 1,
                "pulse_alignment": 1,
                "open_pulse": True,
                "supports_midcircuit_measurement": False,
                "supports_dynamic_circuits": False,
                "dt": 2.2222222222222221e-10,
                "gate_durations": sorted(
                    [
                        ["cx", [0, 1], 4.124444444444444e-07, "s"],
                        ["cx", [1, 0], 3.7688888888888884e-07, "s"],
                        ["cx", [1, 2], 2.702222222222222e-07, "s"],
                        ["cx", [2, 1], 3.0577777777777775e-07, "s"],
                        ["ecr", [1, 0], 3.413333333333333e-07, "s"],
                        ["ecr", [1, 2], 2.3466666666666665e-07, "s"],
                        ["id", [0], 3.5555555555555554e-08, "s"],
                        ["id", [1], 3.5555555555555554e-08, "s"],
                        ["id", [2], 3.5555555555555554e-08, "s"],
                        ["measure", [0], 3.022222222222222e-06, "s"],
                        ["measure", [1], 3.022222222222222e-06, "s"],
                        ["measure", [2], 3.022222222222222e-06, "s"],
                        ["reset", [0], 3.431111111111111e-06, "s"],
                        ["reset", [1], 3.431111111111111e-06, "s"],
                        ["reset", [2], 3.431111111111111e-06, "s"],
                        ["rz", [0], 0.0, "s"],
                        ["rz", [1], 0.0, "s"],
                        ["rz", [2], 0.0, "s"],
                        ["sx", [0], 3.5555555555555554e-08, "s"],
                        ["sx", [1], 3.5555555555555554e-08, "s"],
                        ["sx", [2], 3.5555555555555554e-08, "s"],
                        ["x", [0], 3.5555555555555554e-08, "s"],
                        ["x", [1], 3.5555555555555554e-08, "s"],
                        ["x", [2], 3.5555555555555554e-08, "s"],
                        ["gr", [0, 1, 2, 3], 1.5e-5, "s"],
                    ]
                ),
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

        if api_version == "v0.2.0":
            self._client = MockSuperstaqClient(
                client_name="qiskit-superstaq",
                remote_host=remote_host,
                api_key=api_key,
                api_version=api_version,
                max_retry_seconds=max_retry_seconds,
                verbose=verbose,
            )
        else:
            self._client = gss.superstaq_client._SuperstaqClientV3(
                client_name="qiskit-superstaq",
                remote_host=remote_host,
                api_key=api_key,
                api_version=api_version,
                max_retry_seconds=max_retry_seconds,
                verbose=verbose,
            )


@pytest.fixture
def fake_superstaq_provider() -> MockSuperstaqProvider:
    """Fixture that retrieves the `SuperstaqProvider`.

    Returns:
        The Mock Superstaq provider.
    """
    return MockSuperstaqProvider(api_key="MY_TOKEN")


@pytest.fixture
def fake_superstaq_providerV3() -> MockSuperstaqProvider:
    """Fixture that retrieves the `SuperstaqProvider`.

    Returns:
        The Mock Superstaq provider.
    """
    return MockSuperstaqProvider(api_key="MY_TOKEN", api_version="v0.3.0")
