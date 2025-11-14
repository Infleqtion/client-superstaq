"""API client for collecting machine tasks from Superstaq server."""

from __future__ import annotations

import uuid
from collections.abc import Collection

import general_superstaq as gss


class MachineAPI(gss.superstaq_client._BaseSuperstaqClient):
    """A client for superstaq target workers. Contains the logic to retrieve circuits, return
    results to the server and update the machine specs in the target database.
    """

    def __init__(
        self,
        worker_api_token: str,
        remote_host: str | None = None,
        api_version: str = "v0.3.0",
        circuit_type: gss.models.CircuitType = gss.models.CircuitType.CIRQ,
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
    ) -> None:
        super().__init__(
            client_name="MachineAPI",
            api_key=worker_api_token,
            remote_host=remote_host,
            api_version=api_version,
            circuit_type=circuit_type,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def _task_status_route(self, task_id: uuid.UUID | str) -> str:
        return f"/cq_worker/circuit_status/{task_id}"

    def get_next_circuit(self) -> gss.models.MachineTask | None:
        """Get next circuit for this machine from Superstaq. If no circuit is found, then
        return None.
        """
        response = self.get_request(
            "/cq_worker/next_circuit", query={"circuit_language": self.circuit_type.value}
        )
        if not response:
            return None
        next_circuit = gss.models.MachineTask(**response)
        return next_circuit

    def get_task_status(self, task_id: uuid.UUID | str) -> gss.models.CircuitStatus:
        """Get the status of a task from Superstaq.
        This allows the hardware to query if a user has canceled the task.
        """
        response = self.get_request(self._task_status_route(task_id))
        circuit_status_response = gss.models.MachineTaskStatus(**response)
        circuit_status = circuit_status_response.status
        return circuit_status

    def post_task_status(
        self,
        task_id: uuid.UUID | str,
        status: gss.models.CircuitStatus,
        status_message: str | None = None,
    ) -> None:
        """Post the status of a task to Superstaq."""
        self.post_result(task_id, status, bitstrings=None, status_message=status_message)

    def post_result(
        self,
        task_id: uuid.UUID | str,
        status: gss.models.CircuitStatus,
        bitstrings: Collection[str] | None,
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

        results = gss.models.MachineTaskResults(
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
