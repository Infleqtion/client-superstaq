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
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: Qiskit QuantumCircuit(s) to compile.
            target: String of target AQT device.
            atol: Tolerance to use for approximate gate synthesis (currently just for qutrit gates).
            kwargs: Other desired aqt_compile options.
        Returns:
            Object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        if not target.startswith("aqt_"):
            raise ValueError(f"{target} is not an AQT target")

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

        return qss.compiler_output.read_json_aqt(json_dict, circuits_is_list)

    def aqt_compile_eca(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        num_equivalent_circuits: int,
        random_seed: Optional[int] = None,
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to target AQT device with Equivalent Circuit Averaging
        (ECA).

        See arxiv.org/pdf/2111.04572.pdf for a description of ECA.

        Args:
            circuits: Qiskit QuantumCircuit(s) to compile.
            num_equivalent_circuits: Number of logically equivalent random circuits to generate for
                each input circuit.
            random_seed: Optional seed for circuit randomizer.
            target: String of target AQT device.
            atol: Tolerance to use for approximate gate synthesis (currently just for qutrit gates).
            kwargs: Other desired aqt_compile_eca options.
        Returns:
            Object whose .circuits attribute is a list (or list of lists) of logically equivalent
                QuantumCircuit(s).

            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the QuantumCircuits and the .pulse_lists attribute is
            the list(s) of cycles.
        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        if not target.startswith("aqt_"):
            raise ValueError(f"{target} is not an AQT target")

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
            json_dict, circuits_is_list, num_equivalent_circuits
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

        if not target.startswith("ibmq_"):
            raise ValueError(f"{target} is not an IBMQ target")

        serialized_circuits = qss.serialization.serialize_circuits(circuits)

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(kwargs),
        }

        json_dict = self._client.ibmq_compile(request_json)
        compiled_circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        pulses = gss.serialization.deserialize(json_dict["pulses"])
        final_logical_to_physicals: List[Dict[int, int]] = list(
            map(dict, json.loads(json_dict["final_logical_to_physicals"]))
        )

        if isinstance(circuits, qiskit.QuantumCircuit):
            return qss.compiler_output.CompilerOutput(
                compiled_circuits[0], final_logical_to_physicals[0], pulse_sequences=pulses[0]
            )
        return qss.compiler_output.CompilerOutput(
            compiled_circuits, final_logical_to_physicals, pulse_sequences=pulses
        )

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        mirror_swaps: bool = True,
        base_entangling_gate: str = "xx",
        target: str = "sandia_qscout_qpu",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
            target: String of target representing target device
            mirror_swaps: If mirror swaps should be used.
            base_entangling_gate: The base entangling gate to use.
            kwargs: Other desired qscout_compile options
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        Raises:
            ValueError: If `target` is not a valid QSCOUT target.
            ValueError: If `base_entangling_gate` is not a valid gate option.
        """
        if not target.startswith("sandia_"):
            raise ValueError(f"{target} is not a QSCOUT target")

        qss.superstaq_backend.validate_target(target)

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
        return qss.compiler_output.read_json_qscout(json_dict, circuits_is_list)

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
        if not target.startswith("cq_"):
            raise ValueError(f"{target} is not a CQ target")

        qss.superstaq_backend.validate_target(target)

        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, qiskit.QuantumCircuit)

        request_json = {
            "qiskit_circuits": serialized_circuits,
            "target": target,
            "options": json.dumps(kwargs),
        }
        json_dict = self._client.cq_compile(request_json)

        return qss.compiler_output.read_json_only_circuits(json_dict, circuits_is_list)

    def supercheq(
        self, files: List[List[int]], num_qubits: int, depth: int
    ) -> Tuple[List[qiskit.QuantumCircuit], npt.NDArray[np.float_]]:
        """Docstring."""
        json_dict = self._client.supercheq(files, num_qubits, depth, "qiskit_circuits")
        circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        fidelities = gss.serialization.deserialize(json_dict["fidelities"])
        return circuits, fidelities
