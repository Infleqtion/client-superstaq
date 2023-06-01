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

    if int(integer_param) <= 0:
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
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the backend's native gateset.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            kwargs: Other desired compile options.

        Returns:
            A CompilerOutput object whose .circuit(s) attribute contains optimized compiled
            circuit(s).

        Raises:
            ValueError: If this backend does not support compilation.
        """
        _validate_qiskit_circuits(circuits)
        if self.name().startswith("ibmq_"):
            return self.ibmq_compile(circuits, **kwargs)
        elif self.name().startswith("aqt_"):
            return self.aqt_compile(circuits, **kwargs)
        elif self.name().startswith("sandia_"):
            return self.qscout_compile(circuits, **kwargs)
        elif self.name().startswith("cq_"):
            return self.cq_compile(circuits, **kwargs)
        raise ValueError(
            f"{self.name()} is not a valid target (currently supports AQT, IBMQ, QSCOUT, CQ)."
        )

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        num_equivalent_circuits: Optional[int] = None,
        random_seed: Optional[int] = None,
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT) at
        Lawrence Berkeley National Laboratory. Also allows using Equivalent Circuit Averaging (ECA).

        See arxiv.org/pdf/2111.04572.pdf for a description of ECA.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            num_equivalent_circuits: Number of logically equivalent random circuits to generate for
                each input circuit.
            random_seed: Optional seed used for approximate synthesis and ECA.
            atol: Tolerance to use for approximate gate synthesis (currently just for qutrit gates).
            kwargs: Other desired compile options.

        Returns:
            Object whose .circuit(s) attribute contains the optimized circuits(s). Alternatively for
            ECA, an Object whose .circuits attribute is a list (or list of lists) of logically
            equivalent circuits If qtrl is installed, the object's .seq attribute is a qtrl Sequence
            object containing pulse sequences for each compiled circuit, and its .pulse_list(s)
            attribute contains the corresponding list(s) of cycles.

        Raises:
            ValueError: If this is not an AQT backend.
        """
        if not self.name().startswith("aqt_"):
            raise ValueError(f"{self.name()} is not a valid AQT target.")

        options: Dict[str, Any] = {**kwargs}
        if num_equivalent_circuits is not None:
            options["num_equivalent_cirucits"] = num_equivalent_circuits
        if random_seed is not None:
            options["random_seed"] = random_seed
        if atol is not None:
            options["atol"] = atol

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(options),
        }
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.aqt_compile(request_json)
        return qss.compiler_output.read_json_aqt(
            json_dict, metadata_of_circuits, circuits_is_list, num_equivalent_circuits
        )

    def ibmq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for IBMQ devices.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            kwargs: Other desired compile options.

        Returns:
            An IBMQ CompilerOutput object whose .circuit(s) attribute is an optimized qiskit
            QuantumCircuit(s).

        Raises:
            ValueError: If this is not an IBMQ backend.
        """
        if not self.name().startswith("ibmq_"):
            raise ValueError(f"{self.name()} is not a valid IBMQ target.")

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(kwargs),
        }
        json_dict = self._provider._client.compile(request_json)
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        for circuit, metadata in zip(compiled_circuits, metadata_of_circuits):
            circuit.metadata = metadata
        pulses = None
        if "pulses" in json_dict:
            pulses = gss.serialization.deserialize(json_dict["pulses"])
        final_logical_to_physicals: List[Dict[int, int]] = list(
            map(dict, json.loads(json_dict["final_logical_to_physicals"]))
        )
        if isinstance(circuits, qiskit.QuantumCircuit):
            pulse_sequence = None if pulses is None else pulses[0]
            return qss.compiler_output.CompilerOutput(
                compiled_circuits[0], final_logical_to_physicals[0], pulse_sequences=pulse_sequence
            )

        return qss.compiler_output.CompilerOutput(
            compiled_circuits,
            final_logical_to_physicals,
            pulse_sequences=pulses,
        )

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        mirror_swaps: bool = True,
        base_entangling_gate: str = "xx",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the QSCOUT trapped-ion testbed at Sandia
        National Laboratories [1].

        Compiled circuits are returned as both `qiskit.QuantumCircuit` objects and corresponding
        Jaqal [2] programs (strings).

        References:
            [1] S. M. Clark et al., *Engineering the Quantum Scientific Computing Open User
                Testbed*, IEEE Transactions on Quantum Engineering Vol. 2, 3102832 (2021).
                https://doi.org/10.1109/TQE.2021.3096480.
            [2] B. Morrison, et al., *Just Another Quantum Assembly Language (Jaqal)*, 2020 IEEE
                International Conference on Quantum Computing and Engineering (QCE), 402-408 (2020).
                https://arxiv.org/abs/2008.08042.

        Args:
            circuits: The circuit(s) to compile.
            mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
            base_entangling_gate: The base entangling gate to use (either "xx" or "zz").
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `qiskit QuantumCircuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If this is not a Sandia backend.
            ValueError: If `base_entangling_gate` is not a valid gate option.
        """
        if not self.name().startswith("sandia_"):
            raise ValueError(f"{self.name()} is not a valid Sandia target.")

        if base_entangling_gate not in ("xx", "zz"):
            raise ValueError("base_entangling_gate must be either 'xx' or 'zz'")

        qss.superstaq_backend.validate_target(self.name())
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        options = {
            **kwargs,
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
        }
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

    def cq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for CQ devices.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            kwargs: Other desired compile options.

        Returns:
            An CQ CompilerOutput object.

        Raises:
            ValueError: If this is not a CQ backend.
        """
        if not self.name().startswith("cq_"):
            raise ValueError(f"{self.name()} is not a valid CQ target.")

        qss.superstaq_backend.validate_target(self.name())
        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": json.dumps(kwargs),
        }
        json_dict = self._provider._client.compile(request_json)
        return qss.compiler_output.read_json_only_circuits(
            json_dict, metadata_of_circuits, circuits_is_list
        )
