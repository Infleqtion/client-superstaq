from __future__ import annotations

import uuid
from unittest import mock

import pytest
import requests

import general_superstaq as gss
from general_superstaq.machine_api import MachineAPI


@mock.patch("requests.Session.get")
def test_get_next_circuit(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("token")

    machine_task = gss.models.WorkerTask(
        circuit_ref=str(uuid.uuid4()), circuit="circuit", shots=10
    )

    response1 = requests.Response()
    response1.status_code = requests.codes.ok
    response1._content = machine_task.model_dump_json().encode()

    response2 = requests.Response()
    response2.status_code = requests.codes.ok
    response2._content = b"null"

    mock_get.side_effect = [response1, response2, response2]

    next_circuit = machine_api.get_next_circuit()
    assert next_circuit == machine_task

    next_circuit = machine_api.get_next_circuit()
    assert next_circuit is None

    next_circuit = machine_api.get_next_circuit()
    assert next_circuit is None


@mock.patch("requests.Session.get")
def test_unaccepted_terms_of_use(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("token")

    mock_get.return_value = requests.Response()
    mock_get.return_value.status_code = requests.codes.unauthorized
    mock_get.return_value._content = (
        b'"You must accept the Terms of Use (superstaq.infleqtion.com/terms_of_use)."'
    )

    with pytest.raises(gss.SuperstaqServerException, match=r"accept the Terms of Use"):
        _ = machine_api.get_next_circuit()


@mock.patch("requests.Session.get")
def test_get_task_status(mock_get: mock.MagicMock) -> None:
    machine_api = MachineAPI("token")

    task_id = str(uuid.uuid4())
    machine_task_status = gss.models.WorkerTaskStatus(
        circuit_ref=task_id,
        status=gss.models.CircuitStatus.RUNNING,
    )

    mock_get.return_value = requests.Response()
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value._content = machine_task_status.model_dump_json().encode()

    status = machine_api.get_task_status(task_id)
    assert status == gss.models.CircuitStatus.RUNNING


@mock.patch("requests.Session.post")
def test_post_task_status(mock_post: mock.MagicMock) -> None:
    machine_api = MachineAPI("token")

    task_id = str(uuid.uuid4())

    machine_api.post_task_status(
        task_id=task_id,
        status=gss.models.CircuitStatus.CANCELLED,
    )
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {
        "circuit_ref": task_id,
        "status": gss.models.CircuitStatus.CANCELLED,
        "status_message": None,
        "successful_shots": None,
        "measurements": None,
    }

    machine_api.post_task_status(
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
    machine_api = MachineAPI("token")
    machine_api.update_target_status(
        status=gss.models.TargetStatus.RETIRED,
    )
    mock_put.assert_called_once()
    assert mock_put.call_args.kwargs["json"] == {"status": "retired"}
