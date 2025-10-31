"""API client for collecting machine jobs from Superstaq server."""

from __future__ import annotations

from collections.abc import Collection

import general_superstaq as gss
import logging


class SuperstaqMachineAPI(gss.superstaq_client.HTTPClient):
    """A client for superstaq target workers. Contains the logic to retrieve circuits, return
    results to the server and update the machine specs in the target database.
    """

    def __init__(
        self,
        worker_api_token: str,
        machine_id: str,
        circuit_language: general_superstaq.models.CircuitType,
        remote_host: str | None = None,
        api_version: str = "v0.3.0",
        max_retry_seconds: float = 60,  # 1 minute
        verbose: bool = False,
    ) -> None:
        super().__init__(
            client_name="SuperstaqMachineAPI",
            api_key=worker_api_token,
            remote_host=remote_host,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )
        self.url += "/cq_worker"
        self._machine_id = machine_id
        self._circuit_language = circuit_language

    @property
    def machine_id(self) -> str:
        return self._machine_id

    def _job_status_route(self, job_id: str) -> str:
        return f"/circuit_status/{job_id}"

    def get_next_circuit(self) -> general_superstaq.models.DeviceJob | None:
        """Get next circuit for this machine from Superstaq. If no circuit is found, then
        return None.
        """
        response = self.get_request(
            "/next_circuit", query={"circuit_language": self._circuit_language.value}
        )
        if not response:
            return None
        next_circuit = general_superstaq.models.DeviceJob(**response)
        return next_circuit

    def get_job_status(self, job_id: str) -> general_superstaq.models.CircuitStatus:
        """Get the status of a job from Superstaq.
        This allows the hardware to query if a user has canceled the job.
        """
        response = self.get_request(url=self._job_status_route(job_id))
        circuit_status_response = general_superstaq.models.CircuitStatusResponse(**response)
        circuit_status = circuit_status_response.status
        return circuit_status

    def post_result(
        self,
        job_id: str,
        job_status: general_superstaq.models.CircuitStatus,
        bitstrings: Collection[str] | None,
        status_message: str = "",
    ) -> None:
        """Post the result of a job to Superstaq. Note the server will perform verification of
        the job result including checking if the number of shots in the bitstring is equal to the
        number requested and if the bitstrings match the set of qubit_readout operations requested.
        """
        compressed_bitstrings: dict[str, list[int]] | None = {} if bitstrings is not None else None
        if bitstrings is not None:
            for idx, bs in enumerate(bitstrings):
                bs_index_list = compressed_bitstrings.setdefault(bs, [])
                bs_index_list.append(idx)

        results = general_superstaq.models.DeviceResults(
            circuit_ref=job_id,
            status=job_status,
            status_message=status_message,
            successful_shots=len(bitstrings) if bitstrings is not None else None,
            measurements=compressed_bitstrings,
        )
        self.post_request("/circuit_results", results.model_dump(mode="json"))

    def update_status(
        self,
        target: str,
        status: general_superstaq.models.TargetStatus,
        **config: object,
    ) -> None:
        """Method to update the current machine status to Superstaq."""
        self.put_request("/target_config", {"status": status, **config})
