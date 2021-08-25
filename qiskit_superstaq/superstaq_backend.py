# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Any, List, Union

import qiskit
import requests

import qiskit_superstaq as qss


class SuperstaQBackend(qiskit.providers.BackendV1):
    def __init__(
        self, provider: "qss.superstaq_provider.SuperstaQProvider", url: str, backend: str
    ) -> None:
        self.url = url
        self._provider = provider
        self.configuration_dict = {
            "backend_name": backend,
            "backend_version": "n/a",
            "n_qubits": -1,
            "basis_gates": None,
            "gates": [],
            "local": False,
            "simulator": False,
            "conditional": False,
            "open_pulse": False,
            "memory": False,
            "max_shots": -1,
            "coupling_map": None,
        }
        super().__init__(
            configuration=qiskit.providers.models.BackendConfiguration.from_dict(
                self.configuration_dict
            ),
            provider=provider,
        )

    @classmethod
    def _default_options(cls) -> qiskit.providers.Options:
        return qiskit.providers.Options(shots=1000)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, qss.superstaq_backend.SuperstaQBackend):
            return False

        return (
            self._provider == other._provider
            and self.configuration_dict == other.configuration_dict
        )

    def run(
        self, circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]], **kwargs: int
    ) -> "qss.superstaq_job.SuperstaQJob":

        if isinstance(circuits, qiskit.QuantumCircuit):
            circuits = [circuits]

        superstaq_json = {
            "qasm_strings": [circuit.qasm() for circuit in circuits],
            "backend": self.name(),
            "shots": kwargs.get("shots"),
            "ibmq_token": kwargs.get("ibmq_token"),
            "ibmq_hub": kwargs.get("ibmq_hub"),
            "ibmq_group": kwargs.get("ibmq_group"),
            "ibmq_project": kwargs.get("ibmq_project"),
            "ibmq_pulse": kwargs.get("ibmq_pulse"),
        }

        headers = {
            "Authorization": self._provider.get_access_token(),
            "Content-Type": "application/json",
        }

        res = requests.post(
            self.url + "/" + qss.API_VERSION + "/qasm_strings_multi_job",
            json=superstaq_json,
            headers=headers,
            verify=(self.url == qss.API_URL),
        )

        res.raise_for_status()
        response = res.json()
        if "job_ids" not in response:
            raise Exception

        #  we make a virtual job_id that aggregates all of the individual jobs
        # into a single one, that comma-separates the individual jobs:
        job_id = ",".join(response["job_ids"])
        job = qss.superstaq_job.SuperstaQJob(self, job_id)

        return job
