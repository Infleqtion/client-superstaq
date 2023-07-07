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

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import general_superstaq as gss
import numpy as np
import numpy.typing as npt
import qiskit

import qiskit_superstaq as qss


class SuperstaqProvider(qiskit.providers.ProviderV1, gss.service.Service):
    """Provider for Superstaq backend.

    Typical usage is:

    .. code-block:: python

        import qiskit_superstaq as qss

        ss_provider = qss.SuperstaqProvider('MY_TOKEN')

        backend = ss_provider.get_backend('target')

    where `MY_TOKEN` is the access token provided by Superstaq,
    and `target` is the name of the desired backend.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        """Initializes a SuperstaqProvider.

        Args:
            api_key: A string that allows access to the Superstaq API. If no key is provided, then
                this instance tries to use the environment variable `SUPERSTAQ_API_KEY`. If
                `SUPERSTAQ_API_KEY` is not set, then this instance checks for the
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
            api_version: The version of the API.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            EnvironmentError: If an API key was not provided and could not be found.
        """
        self._name = "superstaq_provider"

        self._client = gss.superstaq_client._SuperstaqClient(
            client_name="qiskit-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return f"<SuperstaqProvider {self._name}>"

    def __repr__(self) -> str:
        return f"<SuperstaqProvider(api_key={self._client.api_key}, name={self._name})>"

    def get_backend(self, target: str) -> qss.SuperstaqBackend:
        """Returns a Superstaq backend.

        Args:
            target: A string containing the name of a target backend.

        Returns:
            A Superstaq backend.
        """
        return qss.SuperstaqBackend(provider=self, target=target)

    def backends(self) -> List[qss.SuperstaqBackend]:
        """Lists the backends available from this provider.

        Returns:
            A list of Superstaq backends.
        """
        targets = self._client.get_targets()["superstaq_targets"]
        backends = []
        for target in targets["compile-and-run"]:
            backends.append(self.get_backend(target))
        return backends

    def resource_estimate(
        self, circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]], target: str
    ) -> Union[gss.ResourceEstimate, List[gss.ResourceEstimate]]:
        """Generates resource estimates for qiskit circuit(s).

        Args:
            circuits: The circuit(s) used during resource estimation.
            target: A string containing the name of a target backend.

        Returns:
            ResourceEstimate(s) containing resource costs (after compilation) for running circuit(s)
            on a backend.
        """
        return self.get_backend(target).resource_estimate(circuits)

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        target: str = "aqt_keysight_qpu",
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
            target: A string containing the name of a target AQT backend.
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
            ValueError: If `target` is not a valid AQT target.
        """
        if not target.startswith("aqt_"):
            raise ValueError(f"{target!r} is not a valid AQT target.")

        return self.get_backend(target).compile(
            circuits,
            num_eca_circuits=num_eca_circuits,
            random_seed=random_seed,
            atol=atol,
            gate_defs=gate_defs,
            **kwargs,
        )

    def aqt_compile_eca(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        num_equivalent_circuits: int,
        random_seed: Optional[int] = None,
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        gate_defs: Optional[Mapping[str, Union[str, npt.NDArray[np.complex_], None]]] = None,
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT) at
        Lawrence Berkeley National Laboratory using Equivalent Circuit Averaging (ECA).

        See arxiv.org/pdf/2111.04572.pdf for a description of ECA.

        Note:
            This method has been deprecated. Instead, use the `num_eca_circuits` argument of
            `aqt_compile()`.

        Args:
            circuits: The circuit(s) to compile.
            num_equivalent_circuits: Number of logically equivalent random circuits to generate for
                each input circuit.
            random_seed: Optional seed for circuit randomizer.
            target: A string containing the name of a target AQT backend.
            atol: An optional tolerance to use for approximate gate synthesis.
            gate_defs: An optional dictionary mapping names in `qtrl` configs to operations, where
                each operation can be either a unitary matrix or None. More specific associations
                take precedence, for example `{"SWAP": <matrix1>, "SWAP/C5C4": <matrix2>}` implies
                `<matrix1>` for all "SWAP" calibrations except "SWAP/C5C4" (which will instead be
                mapped to `<matrix2>` applied to qubits 4 and 5). Setting any calibration to None
                will disable that calibration.
            kwargs: Other desired aqt_compile_eca options.

        Returns:
            Object whose .circuits attribute is a list (or list of lists) of logically equivalent
            circuits. If `qtrl` is installed, the object's .seq attribute is a qtrl Sequence object
            containing pulse sequences for each compiled circuit, and its .pulse_list(s) attribute
            contains the corresponding list(s) of cycles.

        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        warnings.warn(
            "The `aqt_compile_eca()` method has been deprecated, and will be removed in a future "
            "version of qiskit-superstaq. Instead, use the `num_eca_circuits` argument of "
            "`aqt_compile()` to compile circuits for ECA.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.aqt_compile(
            circuits,
            num_eca_circuits=num_equivalent_circuits,
            random_seed=random_seed,
            atol=atol,
            gate_defs=gate_defs,
            **kwargs,
        )

    def ibmq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        target: str = "ibmq_qasm_simulator",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Returns pulse schedule(s) for the given qiskit circuit(s) and target.

        Args:
            circuits: The circuit(s) to compile.
            target: A string containing the name of a target IBMQ backend.
            kwargs: Other desired ibmq_compile options.

        Returns:
            object whose .circuit(s) attribute is an optimized qiskit circuit(s).

        Raises:
            ValueError: If `target` is not a valid IBMQ target.
        """
        if not target.startswith("ibmq_"):
            raise ValueError(f"{target!r} is not a valid IBMQ target.")

        return self.get_backend(target).compile(circuits, **kwargs)

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
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
            target: A string containing the name of a target backend.
            mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
            base_entangling_gate: The base entangling gate to use (either "xx" or "zz").
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `qiskit QuantumCircuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If `target` is not a valid Sandia target.
            ValueError: If `base_entangling_gate` is not a valid gate option.
        """
        if not target.startswith("sandia_"):
            raise ValueError(f"{target!r} is not a valid Sandia target.")

        return self.get_backend(target).compile(
            circuits, mirror_swaps=mirror_swaps, base_entangling_gate=base_entangling_gate, **kwargs
        )

    def cq_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]],
        target: str = "cq_hilbert_qpu",
        **kwargs: Any,
    ) -> qss.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to CQ device, optimized to its native gate set.

        Args:
            circuits: The circuit(s) to compile.
            target: A string containing the name of a target backend.
            kwargs: Other desired cq_compile options.

        Returns:
            object whose .circuit(s) attribute is an optimized qiskit circuit(s).

        Raises:
            ValueError: If `target` is not a valid CQ target.
        """
        if not target.startswith("cq_"):
            raise ValueError(f"{target!r} is not a valid CQ target.")

        return self.get_backend(target).compile(circuits, **kwargs)

    def supercheq(
        self, files: List[List[int]], num_qubits: int, depth: int
    ) -> Tuple[List[qiskit.QuantumCircuit], npt.NDArray[np.float_]]:
        """Returns Supercheq randomly generated circuits and the corresponding fidelity matrices.

        References:
            [1] P. Gokhale et al., *SupercheQ: Quantum Advantage for Distributed Databases*, (2022).
                https://arxiv.org/abs/2212.03850.

        Args:
            files: A list of files specified as binary using ints.
                For example: [[1, 0, 1], [1, 1, 1]].
            num_qubits: The number of qubits to run Supercheq on.
            depth: The depth of the circuits to run Supercheq on.

        Returns:
            A tuple containing a list of `qiskit.QuantumCircuit`s and a list of corresponding
                fidelity matrices.
        """
        json_dict = self._client.supercheq(files, num_qubits, depth, "qiskit_circuits")
        circuits = qss.serialization.deserialize_circuits(json_dict["qiskit_circuits"])
        fidelities = gss.serialization.deserialize(json_dict["fidelities"])
        return circuits, fidelities

    def target_info(self, target: str) -> Dict[str, Any]:
        """Returns information about the device specified by `target`.

        Args:
            target: A string containing the name of a target backend.

        Returns:
            Information about a target backend.
        """
        return self._client.target_info(target)["target_info"]

    def get_targets(self) -> Dict[str, Any]:
        """Gets list of targets.

        Returns:
            A dictionary sorted by "compile-only", "compile-and-run", "unavailable", and "retired".
        """
        return self._client.get_targets()["superstaq_targets"]
