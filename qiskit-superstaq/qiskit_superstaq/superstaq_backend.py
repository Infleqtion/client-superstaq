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
from typing import Any, Dict, List, Optional, Sequence, Union

import general_superstaq as gss
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


def _validate_integer_param(integer_param: object) -> None:
    """Validates that an input parameter is positive and an integer.

    Args:
        integer_param: An input parameter.

    Raises:
        TypeError: If input is not an integer.
        ValueError: If input is negative.
    """

    if not (
        (hasattr(integer_param, "__int__") and int(integer_param) == integer_param)
        or (isinstance(integer_param, str) and integer_param.isdecimal())
    ):
        raise TypeError(f"{integer_param} cannot be safely cast as an integer.")

    if int(integer_param) < 0:
        raise ValueError("Must be a positive integer.")


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
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        num_equivalent_circuits: Optional[int] = 0,
        random_seed: Optional[int] = None,
        atol: Optional[float] = None,
        mirror_swaps: bool = True,
        base_entangling_gate: str = "xx",
        **kwargs: Any,
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
        if self.name().startswith("ibmq_"):
            get_compiler_output = self.get_ibmq_compiler_output
        elif self.name().startswith("aqt_"):
            get_compiler_output = self.get_aqt_compiler_output
        elif self.name().startswith("sandia_"):
            get_compiler_output = self.get_qscout_compiler_output
        elif self.name().startswith("cq_"):
            get_compiler_output = self.get_cq_compiler_output
        else:
            raise ValueError(
                f"{self.name()} is not a valid target (currently supports AQT, IBMQ, QSCOUT)."
            )

        options: Dict[str, Any] = {**kwargs}
        if num_equivalent_circuits is not None:  # aqt eca compile
            _validate_integer_param(num_equivalent_circuits)
            options["num_eca_circuits"] = num_equivalent_circuits
        if random_seed is not None:  # aqt eca compile
            options["random_seed"] = random_seed
        if atol is not None:  # aqt compile
            options["atol"] = atol
        if mirror_swaps is not None:  # qscout compile
            options["mirror_swaps"] = mirror_swaps
        if base_entangling_gate is not None:  # qscout compile
            options["base_entangling_gate"] = base_entangling_gate
        return get_compiler_output(circuits, options)

    def get_aqt_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        options: Optional[Dict[str, Any]] = None,
    ):
        """Gets result of AQT compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An AQT CompilerOutput object.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(options),
        }
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.aqt_compile(request_json)
        num_equivalent_circuits = options["num_eca_circuits"]
        return qss.compiler_output.read_json_aqt(
            json_dict, metadata_of_circuits, circuits_is_list, num_equivalent_circuits
        )

    def get_ibmq_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        options: Optional[Dict[str, Any]] = None,
    ):
        """Gets result of IBMQ compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An IBMQ CompilerOutput object.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(options),
        }
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

    def get_qscout_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        options: Optional[Dict[str, Any]] = None,
    ):
        """Gets result of QSCOUT compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An QSCOUT CompilerOutput object.
        """
        qss.superstaq_backend.validate_target(self.name())
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        base_entangling_gate = options["base_entangling_gate"]
        if base_entangling_gate not in ("xx", "zz"):
            raise ValueError("base_entangling_gate must be either 'xx' or 'zz'")

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(options),
        }
        json_dict = self._provider._client.qscout_compile(request_json)
        return qss.compiler_output.read_json_qscout(
            json_dict, metadata_of_circuits, circuits_is_list
        )

    def get_cq_compiler_output(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        options: Optional[Dict[str, Any]] = None,
    ):
        """Gets result of CQ compilation request.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            request_json: A dictionary storing request information.

        Returns:
            An CQ CompilerOutput object.
        """
        qss.superstaq_backend.validate_target(self.name())
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(options),
        }
        json_dict = self._provider._client.cq_compile(request_json)
        return qss.compiler_output.read_json_only_circuits(
            json_dict, metadata_of_circuits, circuits_is_list
        )
