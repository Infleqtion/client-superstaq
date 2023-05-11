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

import json
import re
from typing import Any, Dict, List, Optional, Union

import qiskit

import qiskit_superstaq as qss


def _validate_qiskit_circuits(circuits: object) -> None:
    """Validates that the input is either a single `qiskit.QuantumCircuit` or a list of
    `qiskit.QuantumCircuit` instances.

    Args:
        circuits: The circuit(s) to run.

    Raises:
        ValueError: If the input is not a `qiskit.QuantumCircuit` or a list of
        `qiskit.QuantumCircuit` instances.

    """
    if not (
        isinstance(circuits, qiskit.QuantumCircuit)
        or (
            isinstance(circuits, Sequence)
            and all(isinstance(circuit, qiskit.QuantumCircuit) for circuit in circuits)
        )
    ):
        raise ValueError(
            "Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a "
            "sequence of `qiskit.QuantumCircuit` instances."
        )


def _get_metadata_of_circuits(
    circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]
) -> List[Dict[Any, Any]]:
    """Extracts metadata from the input qiskit circuit(s).

    Args:
        Circuit(s) from which to extract the metadata.

    Returns:
        A list of dictionaries containing the metadata of the input circuit(s). If a circuit has no
        metadata, an empty dictionary is stored for that circuit.

    """

    metadata_of_circuits = [
        (circuit.metadata or {})
        for circuit in (circuits if isinstance(circuits, list) else [circuits])
    ]

    return metadata_of_circuits


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

    def compile(
        self, circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]], **kwargs: Any
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the backend's native gateset.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
        
        Returns:
            A CompilerOutput object whose .circuit(s) attribute contains optimized compiled
            circuit(s).

        Raises:
            ValueError: If `target` is not a valid AQT or IBMQ target.
        """
        _validate_qiskit_circuits(circuits)
        target = self.name()
        if target.startswith("aqt_"):
            get_compiler_output = self.get_aqt_compiler_output
        elif target.startswith("ibmq_"):
            get_compiler_output = self.get_ibmq_compiler_output
        else:
            raise ValueError(f"{target} is not a valid target (currently supports AQT, IBMQ).")

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(kwargs),
        }
        return get_compiler_output(request_json, circuits)

    def get_aqt_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        request_json: Dict[str, str],
    ):
        """Gets result of AQT compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An AQT CompilerOutput object.
        """
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.aqt_compile(request_json)
        return qss.compiler_output.read_json_aqt(json_dict, metadata_of_circuits, circuits_is_list)

    def get_ibmq_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        request_json: Dict[str, str],
    ):
        """Gets result of IBMQ compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An IBMQ CompilerOutput object.
        """
        json_dict = self._provider._client.ibmq_compile(request_json)
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        for circuit, metadata in zip(compiled_circuits, metadata_of_circuits):
            circuit.metadata = metadata
        pulses = gss.serialization.deserialize(json_dict["pulses"])
        final_logical_to_physicals: List[Dict[int, int]] = list(
            map(dict, json.loads(json_dict["final_logical_to_physicals"]))
        )

        if isinstance(circuits, qiskit.QuantumCircuit):
            return qss.compiler_output.CompilerOutput(
                compiled_circuits[0], final_logical_to_physicals[0], pulse_sequences=pulses[0]
            )
        return qss.compiler_output.CompilerOutput(
            compiled_circuits,
            final_logical_to_physicals,
            pulse_sequences=pulses,
        )
