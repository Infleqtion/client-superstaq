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

import pydantic
import pytest

import general_superstaq as gss

"""This test file will be updated once the new server (v0.3.0) client is implemented.
For now, a single test is sufficient to check that `pydantic` is loading models and working as
expected.
"""


def test_user_token_response() -> None:
    gss.models.UserTokenResponse(
        email="valid.email@infleqtion.com",
        token="abc",
    )
    with pytest.raises(pydantic.ValidationError):
        gss.models.UserTokenResponse(
            email="this is not an email",
            token="abc",
        )


def test_external_provider_credentials() -> None:
    credentials = gss.models.ExternalProviderCredentials()
    assert credentials.cq_token is None
    assert credentials.cq_project_id is None
    assert credentials.cq_org_id is None

    # New-style CQ credentials
    options_new = {"cq_token": "token", "cq_project_id": "123", "cq_org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_new)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"

    # Old-style CQ credentials
    options_old = {"cq_token": {"access_token": "token"}, "project_id": "123", "org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_old)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"


def test_worker_task_results_validation() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match=r"Workers cannot return a status of awaiting_simulation.",
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.AWAITING_SIMULATION,
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=r"Workers cannot return results unless status is COMPLETED",
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.RUNNING,
            successful_shots=10,
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=r"Workers cannot return results unless status is COMPLETED",
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.RUNNING,
            measurements={},
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=(
            r"When status=COMPLETED the worker must return both the measurements and the number of "
            "successful shots."
        ),
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.COMPLETED,
            measurements={},
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=(
            r"When status=COMPLETED the worker must return both the measurements and the number of "
            "successful shots."
        ),
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=10,
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=(r"Measurement keys must be valid bitstrings."),
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=10,
            measurements={"a": [0], "b": [1, 2]},
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=(r"All measurement keys must have the same length."),
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=10,
            measurements={"101": [0], "01": [1, 2]},
        )

    with pytest.raises(
        pydantic.ValidationError,
        match=(r"Not all successful shots have a measurement."),
    ):
        gss.models.WorkerTaskResults(
            circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
            status=gss.models.CircuitStatus.COMPLETED,
            successful_shots=2,
            measurements={"101": [0], "001": [1, 3]},
        )

    _ = gss.models.WorkerTaskResults(
        circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
        status=gss.models.CircuitStatus.COMPLETED,
        successful_shots=4,
        measurements={"000": [0, 1, 3], "101": [2]},
    )

    _ = gss.models.WorkerTaskResults(
        circuit_ref="f76e84f7-0c65-4f0b-b2d7-14135db3900c",
        status=gss.models.CircuitStatus.FAILED,
        status_message="message",
    )
