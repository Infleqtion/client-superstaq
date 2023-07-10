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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit

import qiskit_superstaq as qss


class SuperstaqBackend(qiskit.providers.BackendV1):
    """This class represents a Superstaq backend."""

    def __init__(self, provider: qss.SuperstaqProvider, target: str) -> None:
        """Initializes a SuperstaqBackend.

        Args:
            provider: Provider for a Superstaq backend.
            target: A string containing the name of a target backend.
        """
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

        gss.validation.validate_target(target)

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
        if not isinstance(other, qss.SuperstaqBackend):
            return False

        return (
            self._provider == other._provider
            and self.configuration_dict == other.configuration_dict
        )

    def run(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        shots: int,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> qss.SuperstaqJob:
        """Runs circuits on the stored Superstaq backend.

        Args:
            circuits: A list of circuits to run.
            shots: The number of execution shots (times to run the circuit).
            method:  An optional string that describes the execution method
                (e.g. 'dry-run', 'statevector', etc.).
            kwargs: Other optimization and execution parameters.

        Returns:
            A Superstaq job storing ID and other related info.

        Raises:
            ValueError: If `circuits` contains invalid circuits for submission.
        """
        qss.validation.validate_qiskit_circuits(circuits)
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
            **kwargs,
        )

        #  we make a virtual job_id that aggregates all of the individual jobs
        # into a single one, that comma-separates the individual jobs:
        job_id = ",".join(result["job_ids"])
        job = qss.SuperstaqJob(self, job_id)

        return job

    def retrieve_job(self, job_id: str) -> qss.SuperstaqJob:
        """Gets a job that has been created on the Superstaq API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `qss.SuperstaqJob` which can be queried for status or results.

        Raises:
            SuperstaqNotFoundException: If there was no job with the given `job_id`.
            SuperstaqException: If there was an error accessing the API.
        """
        return qss.SuperstaqJob(self, job_id)

    def compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the backend's native gateset.

        Args:
            circuits: The qiskit QuantumCircuit(s) to compile.
            kwargs: Other desired compile options.

        Returns:
            A `CompilerOutput` object whose .circuit(s) attribute contains optimized compiled
            circuit(s).

        Raises:
            ValueError: If this backend does not support compilation.
        """
        if self.name().startswith("ibmq_"):
            return self.ibmq_compile(circuits, **kwargs)

        elif self.name().startswith("aqt_"):
            return self.aqt_compile(circuits, **kwargs)

        elif self.name().startswith("sandia_"):
            return self.qscout_compile(circuits, **kwargs)

        elif self.name().startswith("cq_"):
            return self.cq_compile(circuits, **kwargs)

        request_json = self._get_compile_request_json(circuits, **kwargs)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.compile(request_json)
        return qss.compiler_output.read_json_only_circuits(json_dict, circuits_is_list)

    def _get_compile_request_json(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> Dict[str, str]:
        qss.validation.validate_qiskit_circuits(circuits)
        gss.validation.validate_target(self.name())

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
        }
        if kwargs:
            request_json["options"] = qss.serialization.to_json(kwargs)
        return request_json

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        *,
        num_eca_circuits: Optional[int] = None,
        random_seed: Optional[int] = None,
        atol: Optional[float] = None,
        gate_defs: Optional[Mapping[str, Union[str, npt.NDArray[np.complex_], None]]] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT).

        AQT is a superconducting transmon quantum computing testbed at Lawrence Berkeley National
        Laboratory. More information can be found at https://aqt.lbl.gov.

        Specifying a nonzero value for `num_eca_circuits` enables compilation with Equivalent
        Circuit Averaging (ECA). See https://arxiv.org/abs/2111.04572 for a description of ECA.

        Args:
            circuits: The circuit(s) to compile.
            num_eca_circuits: Optional number of logically equivalent random circuits to generate
                from each input circuit for Equivalent Circuit Averaging (ECA).
            random_seed: Optional seed used for approximate synthesis and ECA.
            atol: An optional tolerance to use for approximate gate synthesis.
            gate_defs: An optional dictionary mapping names in `qtrl` configs to operations, where
                each operation can be either a unitary matrix or None. More specific associations
                take precedence, for example `{"SWAP": <matrix1>, "SWAP/C5C4": <matrix2>}` implies
                `<matrix1>` for all "SWAP" calibrations except "SWAP/C5C4" (which will instead be
                mapped to `<matrix2>` applied to qubits 4 and 5). Setting any calibration to None
                will disable that calibration.
            kwargs: Other desired compile options.

        Returns:
            Object whose .circuit(s) attribute contains the optimized circuits(s). Alternatively for
            ECA, an Object whose .circuits attribute is a list (or list of lists) of logically
            equivalent circuits. If `qtrl` is installed, the object's .seq attribute is a qtrl
            Sequence object containing pulse sequences for each compiled circuit, and its
            .pulse_list(s) attribute contains the corresponding list(s) of cycles.

        Raises:
            ValueError: If this is not an AQT backend.
        """
        if not self.name().startswith("aqt_"):
            raise ValueError(f"{self.name()!r} is not a valid AQT target.")

        options: Dict[str, Any] = {**kwargs}
        if num_eca_circuits is not None:
            gss.validation.validate_integer_param(num_eca_circuits)
            options["num_eca_circuits"] = int(num_eca_circuits)
        if random_seed is not None:
            gss.validation.validate_integer_param(random_seed)
            options["random_seed"] = int(random_seed)
        if atol is not None:
            options["atol"] = float(atol)
        if gate_defs is not None:
            options["gate_defs"] = gate_defs

        request_json = self._get_compile_request_json(circuits, **options)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.aqt_compile(request_json)
        return qss.compiler_output.read_json_aqt(json_dict, circuits_is_list, num_eca_circuits)

    def ibmq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
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
            raise ValueError(f"{self.name()!r} is not a valid IBMQ target.")

        request_json = self._get_compile_request_json(circuits, **kwargs)
        json_dict = self._provider._client.compile(request_json)
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])

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
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
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
            ValueError: If `base_entangling_gate` is not a valid entangling basis.
        """
        if not self.name().startswith("sandia_"):
            raise ValueError(f"{self.name()!r} is not a valid Sandia target.")

        if base_entangling_gate not in ("xx", "zz"):
            raise ValueError("base_entangling_gate must be either 'xx' or 'zz'")

        options = {
            **kwargs,
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
        }
        request_json = self._get_compile_request_json(circuits, **options)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.qscout_compile(request_json)
        return qss.compiler_output.read_json_qscout(json_dict, circuits_is_list)

    def cq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
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
            raise ValueError(f"{self.name()!r} is not a valid CQ target.")

        request_json = self._get_compile_request_json(circuits, **kwargs)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.compile(request_json)
        return qss.compiler_output.read_json_only_circuits(json_dict, circuits_is_list)

    def target_info(self) -> Dict[str, Any]:
        """Returns information about this backend.

        Returns:
            A dictionary of target information.
        """
        return self._provider._client.target_info(self.name())["target_info"]

    def resource_estimate(
        self, circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]]
    ) -> Union[gss.ResourceEstimate, List[gss.ResourceEstimate]]:
        """Generates resource estimates for qiskit circuit(s).

        Args:
            circuits: The circuit(s) used during resource estimation.

        Returns:
            ResourceEstimate(s) containing resource costs (after compilation) for running circuit(s)
            on this backend.
        """
        request_json = self._get_compile_request_json(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.resource_estimate(request_json)

        resource_estimates = [
            gss.ResourceEstimate(json_data=resource_estimate)
            for resource_estimate in json_dict["resource_estimates"]
        ]
        if circuits_is_list:
            return resource_estimates
        return resource_estimates[0]
