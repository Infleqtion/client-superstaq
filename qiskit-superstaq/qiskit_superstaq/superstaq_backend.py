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
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import qiskit

import qiskit_superstaq as qss


def validate_target(target: str) -> None:  # pylint: disable=missing-function-docstring
    vendor_prefixes = [
        "aqt",
        "aws",
        "cq",
        "hqs",
        "ibmq",
        "ionq",
        "oxford",
        "quera",
        "rigetti",
        "sandia",
        "ss",
    ]

    target_device_types = ["qpu", "simulator"]

    # Check valid format
    match = re.fullmatch("^([A-Za-z0-9-]+)_([A-Za-z0-9-.]+)_([a-z]+)", target)
    if not match:
        raise ValueError(
            f"{target} does not have a valid string format. "
            "Valid target strings should be in the form: "
            "<provider>_<device>_<type>, e.g. ibmq_lagos_qpu."
        )

    prefix, _, device_type = match.groups()

    # Check valid prefix
    if prefix not in vendor_prefixes:
        raise ValueError(
            f"{target} does not have a valid target prefix. "
            f"Valid target prefixes are: {vendor_prefixes}."
        )

    # Check for valid device type
    if device_type not in target_device_types:
        raise ValueError(
            f"{target} does not have a valid target device type. "
            f"Valid target device types are: {target_device_types}."
        )


class SuperstaQBackend(qiskit.providers.BackendV1):  # pylint: disable=missing-class-docstring
    def __init__(self, provider: qss.SuperstaQProvider, target: str) -> None:
        self._provider = provider
        self.configuration_dict = {
            "backend_name": target,
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

        validate_target(target)

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
        if not isinstance(other, qss.SuperstaQBackend):
            return False

        return (
            self._provider == other._provider
            and self.configuration_dict == other.configuration_dict
        )

    def run(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        shots: int,
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> qss.SuperstaQJob:

        if isinstance(circuits, qiskit.QuantumCircuit):
            circuits = [circuits]

        if not all(circuit.count_ops().get("measure") for circuit in circuits):
            # TODO: only raise if the run method actually requires samples (and not for e.g. a
            # statevector simulation)
            raise ValueError("Circuit has no measurements to sample.")

        qiskit_circuits = qss.serialization.serialize_circuits(circuits)

        result = self._provider._client.create_job(
            serialized_circuits={"qiskit_circuits": qiskit_circuits},
            repetitions=shots,
            target=self.name(),
            method=method,
            options=options,
        )

        #  we make a virtual job_id that aggregates all of the individual jobs
        # into a single one, that comma-separates the individual jobs:
        job_id = ",".join(result["job_ids"])
        job = qss.SuperstaQJob(self, job_id)

        return job
