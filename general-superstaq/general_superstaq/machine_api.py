"""API client for collecting machine jobs from Superstaq server."""

from __future__ import annotations
import general_superstaq as gss
import general_superstaq.models
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
        logger: logging.Logger | None = None,
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
        self._machine_id = machine_id
        self._logger = (
            logger if logger is not None else logging.getLogger("SuperstaqMachineAPILogger")
        )
        self._circuit_language = circuit_language

    @property
    def api_base_url(self) -> str:
        return f"{gss.API_URL}/v0.3.0"

    @property
    def machine_id(self) -> str:
        return self._machine_id

    def get_next_circuit_route(self) -> str:
        return f"/cq_worker/next_circuit"

    def post_result_route(self) -> str:
        return f"/cq_worker/circuit_results"

    def get_job_status_route(self, job_id: str) -> str:
        return f"/cq_worker/circuit_status/{job_id}"

    def machine_status_route(self) -> str:
        return f"/cq_worker/{self.machine_id}/machine_config"

    def get_next_circuit(self) -> general_superstaq.models.DeviceJob | None:
        """Get next circuit for this machine from Superstaq. If no circuit is found, then
        return None.
        """
        self._logger.debug("SuperstaqMachineAPI: Getting next circuit")
        try:
            response = self.get_request(
                self.get_next_circuit_route(),
                query={"circuit_language": self._circuit_language.value},
            )
        except Exception as e:
            self._logger.debug(f"SuperstaqMachineAPI: Get next circuit failed: {e}")
            raise
        job_data = response # .json()
        if not job_data:
            self._logger.info("MachineAPI: Did not receive a new circuit")
            return None
        next_circuit = general_superstaq.models.DeviceJob(**response)
        self._logger.debug(f"MachineAPI: Got next circuit: {next_circuit.circuit_ref}")
        return next_circuit

    def get_job_status(self, job_id: str) -> general_superstaq.models.CircuitStatus:
        """Get the status of a job from Superstaq.
        This allows the hardware to query if a user has canceled the job.
        """
        self._logger.debug(f"SuperstaqMachineAPI: Getting job status for job: {job_id}")
        try:
            response = self.get_request(url=self.get_job_status_route(job_id))
        except Exception as e:
            self._logger.error(f"SuperstaqMachineAPI: Get job status failed for {job_id}: {e}")
            raise
        circuit_status_response = general_superstaq.models.CircuitStatusResponse(**response)
        circuit_status = circuit_status_response.status
        self._logger.debug(
            f"SuperstaqMachineAPI: Got job status for cloud job: {job_id} - {circuit_status}"
        )
        return circuit_status

    def post_result(
        self,
        job_id: str,
        job_status: general_superstaq.models.CircuitStatus,
        bitstrings: list[str] | None,
        status_message: str = "",
    ) -> None:
        """Post the result of a job to Superstaq. Note the server will perform verification of
        the job result including checking if the number of shots in the bitstring is equal to the
        number requested and if the bitstrings match the set of qubit_readout operations requested.
        """
        self._logger.debug(f"SuperstaqMachineAPI: Posting results for job: {job_id}")

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
        try:
            self.post_request(self.post_result_route(), results.model_dump(mode="json"))
        except Exception as e:
            self._logger.error(
                f"SuperstaqMachineAPI: Failed for post results for job: {job_id} - {e}"
            )
            raise
        self._logger.debug(f"SuperstaqMachineAPI: Posted results for job: {job_id}")

    def post_machine_status(
        self,
        machine_status: general_superstaq.models,
        machine_config: dict[str, object] = None,
    ) -> None:
        """Method to update the current machine status to Superstaq."""
        self._logger.debug(
            f"SuperstaqMachineAPI: Updating machine status for machine with id: {self.machine_id}"
        )

        try:
            self.put_request(
                url=self.machine_status_route(),
                json={
                    "status": machine_status,
                    **machine_config,
                },
            )
        except Exception as e:
            self._logger.error(
                "SuperstaqMachineAPI: Failed to update machine status for machine "
                f"with id: {self.machine_id} - {e}"
            )
            raise
        self._logger.debug(
            f"SuperstaqMachineAPI: Updated machine status for machine with id: {self.machine_id}"
        )
