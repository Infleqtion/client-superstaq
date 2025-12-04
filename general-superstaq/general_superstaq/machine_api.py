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
"""API client for collecting machine tasks from Superstaq server."""

from __future__ import annotations

from collections.abc import Collection

import general_superstaq as gss


class MachineAPI(gss.superstaq_client._BaseSuperstaqClient):
    """API for machine workers."""

    def _task_status_route(self, task_id: str) -> str:
        return f"/cq_worker/circuit_status/{task_id}"

    def get_next_task(self) -> gss.models.WorkerTask | None:
        """Get next task for this machine from Superstaq. If no task is found, then return None."""
        response = self.get_request(
            "/cq_worker/next_circuit", query={"circuit_language": self.circuit_type.value}
        )
        if not response:
            return None

        return gss.models.WorkerTask(**response)

    def get_task_status(self, task_id: str) -> gss.models.CircuitStatus:
        """Get the status of a task from Superstaq.
        This allows the hardware to query if a user has canceled the task.
        """
        response = self.get_request(self._task_status_route(task_id))
        circuit_status_response = gss.models.WorkerTaskStatus(**response)
        circuit_status = circuit_status_response.status
        return circuit_status

    def post_result(
        self,
        task_id: str,
        status: gss.models.CircuitStatus,
        bitstrings: Collection[str] | None = None,
        status_message: str | None = None,
    ) -> None:
        """Post the result of a task to Superstaq. Note the server will perform verification of
        the task result including checking if the number of shots in the bitstring is equal to the
        number requested and if the bitstrings match the set of qubit_readout operations requested.
        """
        compressed_bitstrings: dict[str, list[int]] | None = None
        if bitstrings is not None:
            compressed_bitstrings = {}
            for idx, bs in enumerate(bitstrings):
                bs_index_list = compressed_bitstrings.setdefault(bs, [])
                bs_index_list.append(idx)

        results = gss.models.WorkerTaskResults(
            circuit_ref=task_id,
            status=status,
            status_message=status_message,
            successful_shots=len(bitstrings) if bitstrings is not None else None,
            measurements=compressed_bitstrings,
        )
        self.post_request("/cq_worker/circuit_results", results.model_dump(mode="json"))

    def update_target_status(
        self,
        status: gss.models.TargetStatus,
        **config: object,
    ) -> None:
        """Method to update the current machine status to Superstaq."""
        self.put_request("/cq_worker/target_config", {"status": status, **config})
