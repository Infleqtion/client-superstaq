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

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit
from general_superstaq import ResourceEstimate, finance, logistics, superstaq_client, user_config

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


class SuperstaQProvider(
    qiskit.providers.ProviderV1, finance.Finance, logistics.Logistics, user_config.UserConfig
):
    """Provider for SuperstaQ backend.

    Typical usage is:

    .. code-block:: python

        import qiskit_superstaq as qss

        ss_provider = qss.SuperstaQProvider('MY_TOKEN')

        backend = ss_provider.get_backend('target')

    where `'MY_TOKEN'` is the access token provided by SuperstaQ,
    and 'target' is the name of the desired backend.

    Args:
        api_key: A string that allows access to the SuperstaQ API. If no key is provided, then
            this instance tries to use the environment variable `SUPERSTAQ_API_KEY`. If
            furthermore that environment variable is not set, then this instance checks for the
            following files:
            - `$XDG_DATA_HOME/super.tech/superstaq_api_key`
            - `$XDG_DATA_HOME/coldquanta/superstaq_api_key`
            - `~/.super.tech/superstaq_api_key`
            - `~/.coldquanta/superstaq_api_key`
            If one of those files exists, then it is treated as a plain text file, and the first
            line of this file is interpreted as an API key.  Failure to find an API key raises
            an `EnvironmentError`.
        remote_host: The location of the API in the form of a URL. If this is None,
            then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
            If that variable is not set, then this uses
            `https://superstaq.super.tech/{api_version}`,
            where `{api_version}` is the `api_version` specified below.
        api_version: Version of the API.
        max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
        verbose: Whether to print to stdio and stderr on retriable errors.
    Raises:
        EnvironmentError: If an API key was not provided and could not be found.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        self._name = "superstaq_provider"

        self._client = superstaq_client._SuperstaQClient(
            client_name="qiskit-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return f"<SuperstaQProvider {self._name}>"

    def __repr__(self) -> str:
        repr1 = f"<SuperstaQProvider(api_key={self._client.api_key}, "
        return repr1 + f"name={self._name})>"

    def get_backend(self, target: str) -> qss.SuperstaQBackend:
        return qss.SuperstaQBackend(provider=self, target=target)

    def backends(self) -> List[qss.SuperstaQBackend]:
        targets = self._client.get_targets()["superstaq_targets"]
        backends = []
        for target in targets["compile-and-run"]:
            backends.append(self.get_backend(target))
        return backends

    def resource_estimate(
        self, circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]], target: str
    ) -> Union[ResourceEstimate, List[ResourceEstimate]]:
        """Generates resource estimates for circuit(s).

        Args:
            circuits: qiskit QuantumCircuit(s).
            target: string of target representing target device
        Returns:
            ResourceEstimate(s) containing resource costs (after compilation)
            for running circuit(s) on target.
        """
        _validate_qiskit_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuit_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
        }

        json_dict = self._client.resource_estimate(request_json)

        resource_estimates = [
            ResourceEstimate(json_data=resource_estimate)
            for resource_estimate in json_dict["resource_estimates"]
        ]
        if circuit_is_list:
            return resource_estimates
        return resource_estimates[0]

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT) at
        Lawrence Berkeley National Laboratory.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target AQT device.
            atol: An optional tolerance to use for approximate gate synthesis.
            kwargs: Other desired aqt_compile options.

        Returns:
            Object whose .circuit(s) attribute contains the optimized circuits(s). If qtrl is
            installed, the object's .seq attribute is a qtrl Sequence object containing pulse
            sequences for each compiled circuit, and its .pulse_list(s) attribute contains the
            corresponding list(s) of cycles.

        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        _validate_qiskit_circuits(circuits)

        if not target.startswith("aqt_"):
            raise ValueError(f"{target} is not an AQT target")

        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        options_dict: Dict[str, Any] = {**kwargs}
        if atol is not None:
            options_dict["atol"] = atol

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(options_dict),
        }

        json_dict = self._client.post_request("/aqt_compile", request_json)

        return qss.compiler_output.read_json_aqt(json_dict, metadata_of_circuits, circuits_is_list)

    def aqt_compile_eca(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        num_equivalent_circuits: int,
        random_seed: Optional[int] = None,
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT) at
        Lawrence Berkeley National Laboratory using Equivalent Circuit Averaging (ECA).

        See arxiv.org/pdf/2111.04572.pdf for a description of ECA.

        Args:
            circuits: The circuit(s) to compile.
            num_equivalent_circuits: Number of logically equivalent random circuits to generate for
                each input circuit.
            random_seed: Optional seed for circuit randomizer.
            target: String of target AQT device.
            atol: An optional tolerance to use for approximate gate synthesis.
            kwargs: Other desired aqt_compile_eca options.

        Returns:
            Object whose .circuits attribute is a list (or list of lists) of logically equivalent
            circuits. If qtrl is installed, the object's .seq attribute is a qtrl Sequence object
            containing pulse sequences for each compiled circuit, and its .pulse_list(s) attribute
            contains the corresponding list(s) of cycles.

        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        _validate_qiskit_circuits(circuits)
        _validate_integer_param(num_equivalent_circuits)
        if not target.startswith("aqt_"):
            raise ValueError(f"{target} is not an AQT target")

        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        options_dict: Dict[str, Union[int, float]] = {
            "num_eca_circuits": num_equivalent_circuits,
            **kwargs,
        }
        if random_seed is not None:
            options_dict["random_seed"] = random_seed
        if atol is not None:
            options_dict["atol"] = atol

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(options_dict),
        }

        json_dict = self._client.post_request("/aqt_compile", request_json)
        return qss.compiler_output.read_json_aqt(
            json_dict, metadata_of_circuits, circuits_is_list, num_equivalent_circuits
        )

    def ibmq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "ibmq_qasm_simulator",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Returns pulse schedule(s) for the given circuit(s) and target.
        Args:
            circuits: Qiskit QuantumCircuit(s)
            target: String of target IBMQ device.
            kwargs: Other desired ibmq_compile options.
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
        Raises:
            ValueError: If `target` is not a valid IBMQ target.
        """

        _validate_qiskit_circuits(circuits)
        if not target.startswith("ibmq_"):
            raise ValueError(f"{target} is not an IBMQ target")

        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(kwargs),
        }

        json_dict = self._client.compile(request_json)
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
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
        mirror_swaps: bool = False,
        base_entangling_gate: str = "xx",
        target: str = "sandia_qscout_qpu",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the QSCOUT trapped-ion testbed at
        Sandia National Laboratories [1].

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
            target: String of target representing target device
            mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
            base_entangling_gate: The base entangling gate to use (either "xx" or "zz").
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `qiskit QuantumCircuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If `target` is not a valid QSCOUT target.
            ValueError: If `base_entangling_gate` is not a valid gate option.
        """
        _validate_qiskit_circuits(circuits)
        if not target.startswith("sandia_"):
            raise ValueError(f"{target} is not a QSCOUT target")

        qss.superstaq_backend.validate_target(target)

        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        if base_entangling_gate not in ("xx", "zz"):
            raise ValueError("base_entangling_gate must be either 'xx' or 'zz'")

        options_dict = {
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
            **kwargs,
        }
        json_dict = self._client.qscout_compile(
            {
                "qiskit_circuits": serialized_circuits,
                "target": target,
                "options": json.dumps(options_dict),
            }
        )
        return qss.compiler_output.read_json_qscout(
            json_dict, metadata_of_circuits, circuits_is_list
        )

    def cq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "cq_hilbert_qpu",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to CQ device, optimized to its native gate set.

        Args:
            circuits: Qiskit QuantumCircuit(s)
            target: String of target representing target device
            kwargs: Other desired cq_compile options.
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
        Raises:
            ValueError: If `target` is not a valid CQ target.
        """
        _validate_qiskit_circuits(circuits)
        if not target.startswith("cq_"):
            raise ValueError(f"{target} is not a CQ target")

        qss.superstaq_backend.validate_target(target)

        metadata_of_circuits = _get_metadata_of_circuits(circuits)
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(kwargs),
        }
        json_dict = self._client.compile(request_json)

        return qss.compiler_output.read_json_only_circuits(
            json_dict, metadata_of_circuits, circuits_is_list
        )

    def supercheq(self, files: List[List[int]], num_qubits: int, depth: int,
    ) -> Tuple[List[qiskit.QuantumCircuit], npt.NDArray[np.float_]]:
        """Returns the randomly generated circuits and the fidelity matrix for inputted 
           files."""
        _validate_integer_param(num_qubits)
        _validate_integer_param(depth)
        json_dict = self._client.supercheq(files, num_qubits, depth, "qiskit_circuits")
        circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        fidelities = gss.serialization.deserialize(json_dict["fidelities"])
        return circuits, fidelities

    def target_info(self, target: str,) -> Dict[str, Any]:
        """Returns information about device specified by `target`."""
        return self._client.target_info(target)
