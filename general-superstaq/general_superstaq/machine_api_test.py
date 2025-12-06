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

import uuid
from unittest import mock

import pytest
import requests

import general_superstaq as gss
from general_superstaq.machine_api import MachineAPI


@mock.patch("requests.Session.get")
def test_get_next_task(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("machine_api", api_key="token")

    task_id = str(uuid.uuid4())
    worker_task = gss.models.WorkerTask(circuit_ref=task_id, circuit='["circuit"]', shots=10)

    response1 = requests.Response()
    response1.status_code = requests.codes.ok
    response1._content = worker_task.model_dump_json().encode()

    response2 = requests.Response()
    response2.status_code = requests.codes.ok
    response2._content = b"null"

    mock_get.side_effect = [response1, response2, response2]

    next_circuit = machine_api.get_next_task()
    assert next_circuit == worker_task

    next_circuit = machine_api.get_next_task()
    assert next_circuit is None

    next_circuit = machine_api.get_next_task()
    assert next_circuit is None


@mock.patch("requests.Session.get")
def test_unaccepted_terms_of_use(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("machine_api", api_key="token")

    mock_get.return_value = requests.Response()
    mock_get.return_value.status_code = requests.codes.unauthorized
    mock_get.return_value._content = (
        b'"You must accept the Terms of Use (superstaq.infleqtion.com/terms_of_use)."'
    )

    with pytest.raises(gss.SuperstaqServerException, match=r"accept the Terms of Use"):
        _ = machine_api.get_next_task()


@mock.patch("requests.Session.get")
def test_get_task_status(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("machine_api", api_key="token")

    task_id = str(uuid.uuid4())
    worker_task_status = gss.models.WorkerTaskStatus(
        circuit_ref=task_id,
        status=gss.models.CircuitStatus.RUNNING,
    )

    mock_get.return_value = requests.Response()
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value._content = worker_task_status.model_dump_json().encode()

    status = machine_api.get_task_status(task_id)
    assert status == gss.models.CircuitStatus.RUNNING


@mock.patch("requests.Session.post")
def test_post_result(mock_post: mock.MagicMock) -> None:
    machine_api = MachineAPI("machine_api", api_key="token")

    task_id = str(uuid.uuid4())

    machine_api.post_result(
        task_id=task_id,
        status=gss.models.CircuitStatus.FAILED,
    )
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {
        "circuit_ref": task_id,
        "status": gss.models.CircuitStatus.FAILED,
        "status_message": None,
        "successful_shots": None,
        "measurements": None,
    }

    machine_api.post_result(
        task_id=task_id,
        status=gss.models.CircuitStatus.FAILED,
        status_message="foo",
    )
    assert mock_post.call_args.kwargs["json"] == {
        "circuit_ref": task_id,
        "status": gss.models.CircuitStatus.FAILED,
        "status_message": "foo",
        "successful_shots": None,
        "measurements": None,
    }

    machine_api.post_result(
        task_id=task_id,
        status=gss.models.CircuitStatus.COMPLETED,
        bitstrings=["111", "101", "111"],
    )
    assert mock_post.call_args.kwargs["json"] == {
        "circuit_ref": task_id,
        "status": gss.models.CircuitStatus.COMPLETED,
        "status_message": None,
        "successful_shots": 3,
        "measurements": {"111": [0, 2], "101": [1]},
    }


@mock.patch("requests.Session.put")
def test_update_target_status(mock_put: mock.MagicMock) -> None:
    machine_api = MachineAPI("machine_api", api_key="token")
    machine_api.update_target_status(
        status=gss.models.TargetStatus.RETIRED,
    )
    mock_put.assert_called_once()
    assert mock_put.call_args.kwargs["json"] == {"status": "retired"}
