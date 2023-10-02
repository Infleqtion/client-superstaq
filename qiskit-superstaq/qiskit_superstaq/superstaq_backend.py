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

import numbers
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit

import qiskit_superstaq as qss

if TYPE_CHECKING:
    from _typeshed import SupportsItems


class SuperstaqBackend(qiskit.providers.BackendV1):
    """This class represents a Superstaq backend."""

    def __init__(self, provider: qss.SuperstaqProvider, target: str) -> None:
        """Initializes a `SuperstaqBackend`.

        Args:
            provider: Provider for a Superstaq backend.
            target: A string containing the name of a target backend.
        """
        self._provider = provider

        target_info = self._provider._client.target_info(target)["target_info"]
        self.configuration_dict = {
            "backend_name": target,
            "backend_version": "n/a",
            "n_qubits": target_info.get("num_qubits"),
            "basis_gates": target_info.get("native_gate_set"),
            "gates": [],
            "local": False,
            "simulator": False,
            "conditional": False,
            "open_pulse": False,
            "memory": False,
            "max_shots": None,
            "coupling_map": None,
        }
        target_info.pop("target", None)
        target_info.pop("num_qubits", None)
        target_info.pop("native_gate_set", None)

        self.configuration_dict.update(target_info)
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
            SuperstaqServerException: If there was an error accessing the API.
        """
        return qss.SuperstaqJob(self, job_id)

    def compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the backend's native gateset.

        Args:
            circuits: The `qiskit.QuantumCircuit`(s) to compile.
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
        return qss.compiler_output.read_json(json_dict, circuits_is_list)

    def _get_compile_request_json(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        **kwargs: Any,
    ) -> Dict[str, str]:
        qss.validation.validate_qiskit_circuits(circuits)
        gss.validation.validate_target(self.name())

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        options = {**self._provider._client.client_kwargs, **kwargs}
        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": self.name(),
            "options": qss.serialization.to_json(options),
        }
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
        dynamical_decoupling: bool = True,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for IBMQ devices.

        Args:
            circuits: The `qiskit.QuantumCircuit`(s) to compile.
            dynamical_decoupling: Applies dynamical decoupling optimization to circuit(s).
            kwargs: Other desired compile options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled circuits(s), and whose
            .pulse_gate_circuit(s) attribute contains the corresponding pulse schedule(s) (when
            available).

        Raises:
            ValueError: If this is not an IBMQ backend.
        """
        if not self.name().startswith("ibmq_"):
            raise ValueError(f"{self.name()!r} is not a valid IBMQ target.")

        options: Dict[str, Any] = {**kwargs}

        options["dynamical_decoupling"] = dynamical_decoupling
        request_json = self._get_compile_request_json(circuits, **options)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.compile(request_json)
        return qss.compiler_output.read_json(json_dict, circuits_is_list)

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        *,
        mirror_swaps: bool = False,
        base_entangling_gate: str = "xx",
        num_qubits: Optional[int] = None,
        error_rates: Optional[SupportsItems[tuple[int, ...], float]] = None,
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
            base_entangling_gate: The base entangling gate to use ("xx", "zz", "sxx", or "szz").
                Compilation with the "xx" and "zz" entangling bases will use arbitrary
                parameterized two-qubit interactions, while the "sxx" and "szz" bases will only use
                fixed maximally-entangling rotations.
            num_qubits: An optional number of qubits that should be present in the compiled
                circuit(s) and Jaqal program(s) (otherwise this will be determined from the input).
            error_rates: Optional dictionary assigning relative error rates to pairs of physical
                qubits, in the form `{<qubit_indices>: <error_rate>, ...}` where `<qubit_indices>`
                is a tuple physical qubit indices (ints) and `<error_rate>` is a relative error rate
                for gates acting on those qubits (for example `{(0, 1): 0.3, (1, 2): 0.2}`) . If
                provided, Superstaq will attempt to map the circuit to minimize the total error on
                each qubit.
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `qiskit.QuantumCircuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If this is not a Sandia backend.
            ValueError: If `base_entangling_gate` is not a valid entangling basis.
        """
        if not self.name().startswith("sandia_"):
            raise ValueError(f"{self.name()!r} is not a valid Sandia target.")

        base_entangling_gate = base_entangling_gate.lower()
        if base_entangling_gate not in ("xx", "zz", "sxx", "szz"):
            raise ValueError("base_entangling_gate must be 'xx', 'zz', 'sxx', or 'szz'")

        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        options = {
            **kwargs,
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
        }

        if isinstance(circuits, qiskit.QuantumCircuit):
            max_circuit_qubits = circuits.num_qubits
        else:
            max_circuit_qubits = max(c.num_qubits for c in circuits)

        if error_rates is not None:
            error_rates_list = list(error_rates.items())
            options["error_rates"] = error_rates_list

            # Use error rate dictionary to set `num_qubits`, if not already specified
            if num_qubits is None:
                max_index = max(q for qs, _ in error_rates_list for q in qs)
                num_qubits = max_index + 1

        elif num_qubits is None:
            num_qubits = max_circuit_qubits

        gss.validation.validate_integer_param(num_qubits)
        if num_qubits < max_circuit_qubits:
            raise ValueError(f"At least {max_circuit_qubits} qubits are required for this input.")
        options["num_qubits"] = num_qubits

        request_json = self._get_compile_request_json(circuits, **options)
        json_dict = self._provider._client.qscout_compile(request_json)
        return qss.compiler_output.read_json_qscout(json_dict, circuits_is_list)

    def cq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        *,
        grid_shape: Optional[Tuple[int, int]] = None,
        control_radius: float = 1.0,
        stripped_cz_rads: float = 0.0,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for CQ devices.

        Args:
            circuits: The `qiskit.QuantumCircuit`(s) to compile.
            grid_shape: Optional fixed dimensions for the rectangular qubit grid (by default the
                actual qubit layout will be pulled from the hardware provider).
            control_radius: The radius with which qubits remain connected
                (ie 1.0 indicates nearest neighbor connectivity).
            stripped_cz_rads: The angle in radians of the stripped cz gate.
            kwargs: Other desired compile options.

        Returns:
            An CQ `CompilerOutput` object.

        Raises:
            ValueError: If this is not a CQ backend.
        """
        if not self.name().startswith("cq_"):
            raise ValueError(f"{self.name()!r} is not a valid CQ target.")

        request_json = self._get_compile_request_json(
            circuits,
            grid_shape=grid_shape,
            control_radius=control_radius,
            stripped_cz_rads=stripped_cz_rads,
            **kwargs,
        )
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._provider._client.compile(request_json)
        return qss.compiler_output.read_json(json_dict, circuits_is_list)

    def target_info(self) -> Dict[str, Any]:
        """Returns information about this backend.

        Returns:
            A dictionary of target information.
        """
        return self.configuration_dict

    def resource_estimate(
        self, circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]]
    ) -> Union[gss.ResourceEstimate, List[gss.ResourceEstimate]]:
        """Generates resource estimates for qiskit circuit(s).

        Args:
            circuits: The circuit(s) used during resource estimation.

        Returns:
            `ResourceEstimate`(s) containing resource costs (after compilation) for running
            circuit(s) on this backend.
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

    def submit_aces(
        self,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: Optional[str] = None,
        noise: Optional[str] = None,
        error_prob: Optional[Union[float, Tuple[float, float, float]]] = None,
        tag: Optional[str] = None,
        lifespan: Optional[int] = None,
    ) -> str:
        """Submits the jobs to characterize this target through the ACES protocol.

        The following gate eigenvalues are eestimated. For each qubit in the device, we consider
        six Clifford gates. These are given by the XZ maps: XZ, ZX, -YZ, -XY, ZY, YX. For each of
        these gates, three eigenvalues are returned (X, Y, Z, in that order). Then, the two-qubit
        gate considered here is the CZ in linear connectivity (each qubit n with n + 1). For this
        gate, 15 eigenvalues are considered: XX, XY, XZ, XI, YX, YY, YZ, YI, ZX, ZY, ZZ, ZI, IX, IY
        IZ, in that order.

        If n qubits are characterized, the first 18 * n entries of the list returned by
        `process_aces` will contain the  single-qubit eigenvalues for each gate in the order above.
        After all the single-qubit eigenvalues, the next 15 * (n - 1) entries will contain for the
        CZ connections, in ascending order.

        The protocol in detail can be found in: https://arxiv.org/abs/2108.05803.

        Args:
            qubits: A list with the qubit indices to characterize.
            shots: How many shots to use per circuit submitted.
            num_circuits: How many random circuits to use in the protocol.
            mirror_depth: The half-depth of the mirror portion of the random circuits.
            extra_depth: The depth of the fully random portion of the random circuits.
            method: Which type of method to execute the circuits with.
            noise: Noise model to simulate the protocol with. Valid strings are
                "symmetric_depolarize", "phase_flip", "bit_flip" and "asymmetric_depolarize".
            error_prob: The error probabilities if a string was passed to `noise`.
                * For "asymmetric_depolarize", `error_prob` will be a three-tuple with the error
                rates for the X, Y, Z gates in that order. So, a valid argument would be
                `error_prob = (0.1, 0.1, 0.1)`. Notice that these values must add up to less than
                or equal to 1.
                * For the other channels, `error_prob` is one number less than or equal to 1, e.g.,
                `error_prob = 0.1`.
            tag: Tag for all jobs submitted for this protocol.
            lifespan: How long to store the jobs submitted for in days (only works with right
                permissions).

        Returns:
            A string with the job id for the ACES job created.

        Raises:
            ValueError: If the target or noise model is not valid.
            SuperstaqServerException: If the request fails.
        """
        noise_dict: Dict[str, object] = {}
        if noise:
            noise_dict["type"] = noise
            noise_dict["params"] = (
                (error_prob,) if isinstance(error_prob, numbers.Number) else error_prob
            )

        return self._provider._client.submit_aces(
            target=self.name(),
            qubits=qubits,
            shots=shots,
            num_circuits=num_circuits,
            mirror_depth=mirror_depth,
            extra_depth=extra_depth,
            method=method,
            noise=noise_dict,
            tag=tag,
            lifespan=lifespan,
        )

    def process_aces(self, job_id: str) -> List[float]:
        """Process a job submitted through `submit_aces`.

        Args:
            job_id: The job id returned by `submit_aces`.

        Returns:
            The estimated eigenvalues.
        """
        return self._provider._client.process_aces(job_id=job_id)
