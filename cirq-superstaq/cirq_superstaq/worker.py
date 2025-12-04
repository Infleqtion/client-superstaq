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

import dataclasses
from collections.abc import Collection
from typing import TYPE_CHECKING

import general_superstaq as gss
from general_superstaq.machine_api import MachineAPI

import cirq_superstaq as css

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class Task:
    """A single task to be executed on the machine."""

    task_id: str
    circuit: cirq.Circuit
    shots: int
    metadata: dict[str, object]
    user_email: str | None


class Worker:
    """A client for superstaq target workers. Contains the logic to retrieve circuits, return
    results to the server and update the machine specs in the target database.
    """

    def __init__(
        self,
        worker_api_token: str,
        remote_host: str | None = None,
        api_version: str = "v0.3.0",
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
    ) -> None:
        self._client = MachineAPI(
            client_name="CirqWorker",
            api_key=worker_api_token,
            remote_host=remote_host,
            api_version=api_version,
            circuit_type=gss.models.CircuitType.CIRQ,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def get_next_task(self) -> Task | None:
        """Get next task for this machine from Superstaq. If no task is found, then return None."""
        worker_task = self._client.get_next_task()
        if worker_task is None:
            return None

        return Task(
            task_id=worker_task.circuit_ref,
            circuit=css.deserialize_circuits(worker_task.circuit)[0],
            shots=worker_task.shots,
            metadata=worker_task.metadata,
            user_email=worker_task.user_email,
        )

    def get_task_status(self, task_id: str) -> gss.models.CircuitStatus:
        """Get the status of a task from Superstaq.
        This allows the hardware to query if a user has canceled the task.
        """
        return self._client.get_task_status(task_id)

    def post_task_status(
        self,
        task_id: str,
        status: gss.models.CircuitStatus,
        status_message: str | None = None,
    ) -> None:
        """Post the status of a task to Superstaq."""
        self._client.post_result(task_id, status, bitstrings=None, status_message=status_message)

    def post_result(
        self,
        task_id: str,
        bitstrings: Collection[str] | None,
        status_message: str | None = None,
    ) -> None:
        """Post the result of a task to Superstaq. Note the server will perform verification of
        the task result including checking if the number of shots in the bitstring is equal to the
        number requested and if the bitstrings match the set of qubit_readout operations requested.
        """
        self._client.post_result(
            task_id,
            status=gss.models.CircuitStatus.COMPLETED,
            bitstrings=bitstrings,
            status_message=status_message,
        )

    def update_target_status(
        self,
        status: gss.models.TargetStatus,
        **config: object,
    ) -> None:
        """Method to update the current machine status to Superstaq."""
        self._client.update_target_status(status, **config)
