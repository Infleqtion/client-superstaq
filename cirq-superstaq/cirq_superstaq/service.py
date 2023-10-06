# Copyright 2021 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Service to access Superstaqs API."""
from __future__ import annotations

import numbers
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import cirq
import general_superstaq as gss
import numpy as np
import numpy.typing as npt
from general_superstaq import ResourceEstimate, superstaq_client

import cirq_superstaq as css

if TYPE_CHECKING:
    from _typeshed import SupportsItems


def _to_matrix_gate(matrix: npt.ArrayLike) -> cirq.MatrixGate:
    """Convert a unitary matrix into a `cirq.MatrixGate` acting either on qubits or on qutrits.

    Args:
        matrix: The (unitary) matrix to be converted.

    Returns:
        A `cirq.MatrixGate` with the given unitary.

    Raises:
        ValueError: If `matrix` could not be interpreted as a unitary gate acting on either
            qubits or qutrits.
    """

    matrix = np.asarray(matrix, dtype=complex)

    for dimension in (2, 3):
        num_qids = int(round(np.log(matrix.size) / np.log(dimension**2)))
        if matrix.shape == (dimension**num_qids, dimension**num_qids):
            qid_shape = (dimension,) * num_qids
            return cirq.MatrixGate(matrix, qid_shape=qid_shape)

    raise ValueError(
        "Could not determine qid_shape from array shape, consider using a `cirq.MatrixGate` "
        "instead."
    )


def counts_to_results(
    counter: Mapping[str, float],
    circuit: cirq.AbstractCircuit,
    param_resolver: cirq.ParamResolver,
) -> cirq.ResultDict:
    """Converts a `collections.Counter` to a `cirq.ResultDict`.

    Args:
        counter: The `collections.Counter` of counts for the run.
        circuit: The circuit to run.
        param_resolver: A `cirq.ParamResolver` to resolve parameters in `circuit`.

    Returns:
        A `cirq.ResultDict` for the given circuit and counter.

    """
    measurement_key_names = list(circuit.all_measurement_key_names())
    measurement_key_names.sort()
    # Combines all the measurement key names into a string: {'0', '1'} -> "01"
    combine_key_names = "".join(measurement_key_names)

    samples: List[List[int]] = []
    if not all(counts == int(counts) for counts in counter.values()):
        warnings.warn(
            "The raw counts contain fractional values due to measurement error mitigation; please "
            "use service.get_counts to see raw results.",
            stacklevel=2,
        )
    if not all(counts >= 0 for counts in counter.values()):
        warnings.warn(
            "The raw counts contain negative values due to measurement error mitigation; please "
            "use service.get_counts to see raw results.",
            stacklevel=2,
        )
    for key, counts_of_key in counter.items():
        # Combines the keys of the counter into a list. If key = "01", keys_as_list = [0, 1]
        keys_as_list: List[int] = list(map(int, key))

        # Gets execution counts per bitstring, e.g., collections.Counter({"01": 48, "11": 52})["01"]
        # = 48. Per execution shot, appends bitstring to `samples` list. E.g., if counter is
        # collections.Counter({"01": 48, "11": 52}), [0, 1] is appended 48 times and [1, 1] is
        # appended 52 times.
        counts_of_key = round(counts_of_key)
        for _ in range(counts_of_key):
            samples.append(keys_as_list)

    result = cirq.ResultDict(
        params=param_resolver,
        measurements={
            combine_key_names: np.array(samples),
        },
    )

    return result


class Service(gss.service.Service):
    """A class to access Superstaq's API.

    To access the API, this class requires a remote host url and an API key. These can be
    specified in the constructor via the parameters `remote_host` and `api_key`. Alternatively
    these can be specified by setting the environment variables `SUPERSTAQ_REMOTE_HOST` and
    `SUPERSTAQ_API_KEY`, or setting an API key in a configuration file.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        remote_host: Optional[str] = None,
        default_target: Optional[str] = None,
        api_version: str = gss.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
        cq_token: Optional[str] = None,
        ibmq_token: Optional[str] = None,
        ibmq_instance: Optional[str] = None,
        ibmq_channel: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        """Creates the Service to access Superstaq's API.

        Args:
            api_key: A string that allows access to the Superstaq API. If no key is provided, then
                this instance tries to use the environment variable `SUPERSTAQ_API_KEY`. If
                furthermore that environment variable is not set, then this instance checks for the
                following files:
                - `$XDG_DATA_HOME/super.tech/superstaq_api_key`
                - `$XDG_DATA_HOME/coldquanta/superstaq_api_key`
                - `~/.super.tech/superstaq_api_key`
                - `~/.coldquanta/superstaq_api_key`
                If one of those files exists, the it is treated as a plain text file, and the first
                line of this file is interpreted as an API key.  Failure to find an API key raises
                an `EnvironmentError`.
            remote_host: The location of the api in the form of an url. If this is None,
                then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
                If that variable is not set, then this uses
                `flask-service.cgvd1267imk10.us-east-1.cs.amazonlightsail.com/{api_version}`,
                where `{api_version}` is the `api_version` specified below.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call
            api_version: Version of the api.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.
            cq_token: Token from CQ cloud.This is required to submit circuits to CQ hardware.
            ibmq_token: Your IBM Quantum or IBM Cloud token. This is required to submit circuits
                to IBM hardware, or to access non-public IBM devices you may have access to.
            ibmq_instance: An optional instance to use when running IBM jobs.
            ibmq_channel: The type of IBM account. Must be either "ibm_quantum" or "ibm_cloud".
            kwargs: Other optimization and execution parameters.

        Raises:
            EnvironmentError: If an API key was not provided and could not be found.
        """
        self.default_target = default_target
        self._client = superstaq_client._SuperstaqClient(
            client_name="cirq-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
            cq_token=cq_token,
            ibmq_token=ibmq_token,
            ibmq_instance=ibmq_instance,
            ibmq_channel=ibmq_channel,
            **kwargs,
        )

    def _resolve_target(self, target: Union[str, None]) -> str:
        target = target or self.default_target
        if not target:
            raise ValueError(
                "This call requires a target, but none was provided and default_target is not set."
            )

        gss.validation.validate_target(target)
        return target

    @overload
    def get_counts(
        self,
        circuits: cirq.Circuit,
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, int]:
        ...

    @overload
    def get_counts(
        self,
        circuits: Sequence[cirq.Circuit],
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, int]]:
        ...

    def get_counts(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, int], List[Dict[str, int]]]:
        """Runs circuit(s) on the Superstaq API and returns the result(s) as a `dict`.

        Args:
            circuits: The circuit(s) to run.
            repetitions: The number of times to run the circuit(s).
            target: Where to run the job.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in `circuits`.
            method: Optional execution method.
            kwargs: Other optimization and execution parameters.

        Returns:
            The counts from running the circuit(s).
        """
        resolved_circuits = cirq.resolve_parameters(circuits, param_resolver)
        job = self.create_job(resolved_circuits, int(repetitions), target, method, **kwargs)
        if isinstance(circuits, cirq.Circuit):
            return job.counts(0)
        return [job.counts(i) for i in range(len(circuits))]

    @overload
    def run(
        self,
        circuits: cirq.Circuit,
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> cirq.ResultDict:
        ...

    @overload
    def run(
        self,
        circuits: Sequence[cirq.Circuit],
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> List[cirq.ResultDict]:
        ...

    def run(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[cirq.ResultDict, List[cirq.ResultDict]]:
        """Runs circuit(s) on the Superstaq API and returns the result(s) as `cirq.ResultDict`.

        WARNING: This may return unexpected results when used with measurement error mitigation. Use
        `service.create_job()` or `service.get_counts()` instead.

        Args:
            circuits: The circuit(s) to run.
            repetitions: The number of times to run the circuit(s).
            target: Where to run the job.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in `circuits`.
            method: Execution method.
            kwargs: Other optimization and execution parameters.

        Returns:
            The `cirq.ResultDict` object(s) from running the circuit(s).
        """
        resolved_circuits = cirq.resolve_parameters(circuits, param_resolver)
        job = self.create_job(resolved_circuits, int(repetitions), target, method, **kwargs)

        if isinstance(circuits, cirq.Circuit):
            return counts_to_results(job.counts(0), circuits, param_resolver)
        return [
            counts_to_results(job.counts(i), circuit, param_resolver)
            for i, circuit in enumerate(circuits)
        ]

    def sampler(self, target: Optional[str] = None) -> cirq.Sampler:
        """Returns a `cirq.Sampler` object for accessing sampler interface.

        Args:
            target: Target to sample against.

        Returns:
            A `cirq.Sampler` for the Superstaq API.
        """
        target = self._resolve_target(target)
        return css.sampler.Sampler(service=self, target=target)

    def create_job(
        self,
        circuits: Union[cirq.AbstractCircuit, Sequence[cirq.AbstractCircuit]],
        repetitions: int = 1000,
        target: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> css.job.Job:
        """Creates a new job to run the given circuit(s).

        Args:
            circuits: The circuit or list of circuits to run.
            repetitions: The number of times to repeat the circuit. Defaults to 1000.
            target: Where to run the job.
            method: The optional execution method.
            kwargs: Other optimization and execution parameters.

        Returns:
            A `css.Job` which can be queried for status or results.

        Raises:
            ValueError: If there are no measurements in `circuits`.
            SuperstaqServerException: If there was an error accessing the API.
        """
        css.validation.validate_cirq_circuits(circuits, require_measurements=True)
        serialized_circuits = css.serialization.serialize_circuits(circuits)

        target = self._resolve_target(target)
        result = self._client.create_job(
            serialized_circuits={"cirq_circuits": serialized_circuits},
            repetitions=repetitions,
            target=target,
            method=method,
            **kwargs,
        )
        # Make a virtual job_id that aggregates all of the individual jobs
        # into a single one that comma-separates the individual jobs.
        job_id = ",".join(result["job_ids"])

        # The returned job does not have fully populated fields; they will be filled out by
        # when the new job's status is first queried
        return self.get_job(job_id=job_id)

    def get_job(self, job_id: str) -> css.job.Job:
        """Gets a job that has been created on the Superstaq API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `css.Job` which can be queried for status or results.

        Raises:
            SuperstaqServerException: If there was an error accessing the API.
        """
        return css.job.Job(client=self._client, job_id=job_id)

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
        """Get the querying user's account balance in USD.

        Args:
            pretty_output: Whether to return a pretty string or a float of the balance.

        Returns:
            If `pretty_output` is `True`, returns the balance as a nicely formatted string
            ($-prefix, commas on LHS every three digits, and two digits after period). Otherwise,
            simply returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"${balance:,.2f}"
        return balance

    def get_targets(self) -> Dict[str, List[str]]:
        """Gets a list of available, unavailable, and retired targets.

        Returns:
            A list of Superstaq targets.
        """
        return self._client.get_targets()["superstaq_targets"]

    def resource_estimate(
        self, circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]], target: Optional[str] = None
    ) -> Union[ResourceEstimate, List[ResourceEstimate]]:
        """Generates resource estimates for circuit(s).

        Args:
            circuits: The circuit(s) to generate resource estimate.
            target: String of target representing target device.

        Returns:
            `ResourceEstimate`(s) containing resource costs (after compilation).
        """
        css.validation.validate_cirq_circuits(circuits)
        circuit_is_list = not isinstance(circuits, cirq.Circuit)
        serialized_circuit = css.serialization.serialize_circuits(circuits)

        target = self._resolve_target(target)

        request_json = {
            "cirq_circuits": serialized_circuit,
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

    def aqt_compile_eca(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        num_equivalent_circuits: int,
        random_seed: Optional[int] = None,
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        gate_defs: Optional[
            Mapping[str, Union[npt.NDArray[np.complex_], cirq.Gate, cirq.Operation, None]]
        ] = None,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for AQT using ECA.

        The Advanced Quantum Testbed (AQT) is a superconducting transmon quantum computing testbed
        at Lawrence Berkeley National Laboratory. See arxiv.org/pdf/2111.04572.pdf for a description
        of Equivalent Circuit Averaging (ECA).

        Note:
            This method has been deprecated. Instead, use the `num_eca_circuits` argument of
            `aqt_compile()`.

        Args:
            circuits: The circuit(s) to compile.
            num_equivalent_circuits: Number of logically equivalent random circuits to generate for
                each input circuit.
            random_seed: Optional seed for circuit randomizer.
            target: String of target AQT device.
            atol: An optional tolerance to use for approximate gate synthesis.
            gate_defs: An optional dictionary mapping names in `qtrl` configs to operations, where
                each operation can be a unitary matrix, `cirq.Gate`, `cirq.Operation`, or None. More
                specific associations take precedence, for example `{"SWAP": cirq.SQRT_ISWAP,
                "SWAP/C5C4": cirq.SQRT_ISWAP_INV}` implies `SQRT_ISWAP` for all "SWAP" calibrations
                except "SWAP/C5C4" (which will instead be mapped to a `SQRT_ISWAP_INV` gate on
                qubits 4 and 5). Setting any calibration to None will disable that calibration.
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
            "version of cirq-superstaq. Instead, use the `num_eca_circuits` argument of "
            "`aqt_compile()` to compile circuits for ECA.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.aqt_compile(
            circuits,
            target=target,
            num_eca_circuits=num_equivalent_circuits,
            random_seed=random_seed,
            atol=atol,
            gate_defs=gate_defs,
            **kwargs,
        )

    def aqt_compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str = "aqt_keysight_qpu",
        *,
        num_eca_circuits: Optional[int] = None,
        random_seed: Optional[int] = None,
        atol: Optional[float] = None,
        gate_defs: Optional[
            Mapping[str, Union[npt.NDArray[np.complex_], cirq.Gate, cirq.Operation, None]]
        ] = None,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT).

        AQT is a superconducting transmon quantum computing testbed at Lawrence Berkeley National
        Laboratory. More information can be found at https://aqt.lbl.gov.

        Specifying a nonzero value for `num_eca_circuits` enables compilation with Equivalent
        Circuit Averaging (ECA). See https://arxiv.org/abs/2111.04572 for a description of ECA.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target AQT device.
            num_eca_circuits: Optional number of logically equivalent random circuits to generate
                from each input circuit for Equivalent Circuit Averaging (ECA).
            random_seed: Optional seed used for approximate synthesis and ECA.
            atol: An optional tolerance to use for approximate gate synthesis.
            gate_defs: An optional dictionary mapping names in `qtrl` configs to operations, where
                each operation can be a unitary matrix, `cirq.Gate`, `cirq.Operation`, or None. More
                specific associations take precedence, for example `{"SWAP": cirq.SQRT_ISWAP,
                "SWAP/C5C4": cirq.SQRT_ISWAP_INV}` implies `SQRT_ISWAP` for all "SWAP" calibrations
                except "SWAP/C5C4" (which will instead be mapped to a `SQRT_ISWAP_INV` gate on
                qubits 4 and 5). Setting any calibration to None will disable that calibration.
            kwargs: Other desired compile options.

        Returns:
            Object whose .circuit(s) attribute contains the optimized circuits(s). Alternatively for
            ECA, an object whose .circuits attribute is a list (or list of lists) of logically
            equivalent circuits. If `qtrl` is installed, the object's .seq attribute is a qtrl
            Sequence object containing pulse sequences for each compiled circuit, and its
            .pulse_list(s) attribute contains the corresponding list(s) of cycles.

        Raises:
            ValueError: If `target` is not a valid AQT target.
        """
        target = self._resolve_target(target)
        if not target.startswith("aqt_"):
            raise ValueError(f"{target!r} is not a valid AQT target.")

        css.validation.validate_cirq_circuits(circuits)
        serialized_circuits = css.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)

        request_json = {
            "cirq_circuits": serialized_circuits,
            "target": target,
        }

        options_dict: Dict[str, object]
        options_dict = {**kwargs}

        if num_eca_circuits is not None:
            gss.validation.validate_integer_param(num_eca_circuits)
            options_dict["num_eca_circuits"] = int(num_eca_circuits)
        if random_seed is not None:
            gss.validation.validate_integer_param(random_seed)
            options_dict["random_seed"] = int(random_seed)
        if atol is not None:
            options_dict["atol"] = float(atol)
        if gate_defs is not None:
            gate_defs_cirq = {}
            for key, val in gate_defs.items():
                if val is not None and not isinstance(val, (cirq.Gate, cirq.Operation)):
                    val = _to_matrix_gate(val)
                gate_defs_cirq[key] = val
            options_dict["gate_defs"] = gate_defs_cirq

        if options_dict:
            request_json["options"] = cirq.to_json(options_dict)

        json_dict = self._client.post_request("/aqt_compile", request_json)
        return css.compiler_output.read_json_aqt(json_dict, circuits_is_list, num_eca_circuits)

    def qscout_compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str = "sandia_qscout_qpu",
        *,
        mirror_swaps: bool = False,
        base_entangling_gate: str = "xx",
        num_qubits: Optional[int] = None,
        error_rates: Optional[SupportsItems[tuple[int, ...], float]] = None,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the QSCOUT trapped-ion testbed at
        Sandia National Laboratories [1].

        Compiled circuits are returned as both `cirq.Circuit` objects and corresponding Jaqal [2]
        programs (strings).

        References:
            [1] S. M. Clark et al., *Engineering the Quantum Scientific Computing Open User
                Testbed*, IEEE Transactions on Quantum Engineering Vol. 2, 3102832 (2021).
                https://doi.org/10.1109/TQE.2021.3096480.
            [2] B. Morrison, et al., *Just Another Quantum Assembly Language (Jaqal)*, 2020 IEEE
                International Conference on Quantum Computing and Engineering (QCE), 402-408 (2020).
                https://arxiv.org/abs/2008.08042.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target representing target device.
            mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
            base_entangling_gate: The base entangling gate to use ("xx", "zz", "sxx", or "szz").
                Compilation with the "xx" and "zz" entangling bases will use arbitrary
                parameterized two-qubit interactions, while the "sxx" and "szz" bases will only use
                fixed maximally-entangling rotations.
            num_qubits: An optional number of qubits that should be initialized in the returned
                Jaqal program(s) (by default this will be determined from the input circuits).
            error_rates: Optional dictionary assigning relative error rates to pairs of physical
                qubits, in the form `{<qubit_indices>: <error_rate>, ...}` where `<qubit_indices>`
                is a tuple physical qubit indices (ints) and `<error_rate>` is a relative error rate
                for gates acting on those qubits (for example `{(0, 1): 0.3, (1, 2): 0.2}`) . If
                provided, Superstaq will attempt to map the circuit to minimize the total error on
                each qubit. Omitted qubit pairs are assumed to be error-free.
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `cirq.Circuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If `base_entangling_gate` is not a valid gate option.
            ValueError: If `target` is not a valid Sandia target.
        """
        target = self._resolve_target(target)
        if not target.startswith("sandia_"):
            raise ValueError(f"{target!r} is not a valid Sandia target.")

        base_entangling_gate = base_entangling_gate.lower()
        if base_entangling_gate not in ("xx", "zz", "sxx", "szz"):
            raise ValueError("base_entangling_gate must be 'xx', 'zz', 'sxx', or 'szz'")

        css.validation.validate_cirq_circuits(circuits)
        serialized_circuits = css.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)

        options_dict = {
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
            **kwargs,
        }

        if circuits_is_list:
            max_circuit_qubits = max(cirq.num_qubits(c) for c in circuits)
        else:
            max_circuit_qubits = cirq.num_qubits(circuits)

        if error_rates is not None:
            error_rates_list = list(error_rates.items())
            options_dict["error_rates"] = error_rates_list

            # Use error rate dictionary to set `num_qubits`, if not already specified
            if num_qubits is None:
                max_index = max(q for qs, _ in error_rates_list for q in qs)
                num_qubits = max_index + 1

        elif num_qubits is None:
            num_qubits = max_circuit_qubits

        gss.validation.validate_integer_param(num_qubits)
        if num_qubits < max_circuit_qubits:
            raise ValueError(f"At least {max_circuit_qubits} qubits are required for this input.")
        options_dict["num_qubits"] = num_qubits

        json_dict = self._client.qscout_compile(
            {
                "cirq_circuits": serialized_circuits,
                "options": cirq.to_json(options_dict),
                "target": target,
            }
        )

        return css.compiler_output.read_json_qscout(json_dict, circuits_is_list)

    def cq_compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str = "cq_hilbert_qpu",
        *,
        grid_shape: Optional[Tuple[int, int]] = None,
        control_radius: float = 1.0,
        stripped_cz_rads: float = 0.0,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) to the target CQ device.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target CQ device.
            grid_shape: Optional fixed dimensions for the rectangular qubit grid (by default the
                actual qubit layout will be pulled from the hardware provider).
            control_radius: The radius with which qubits remain connected
                (ie 1.0 indicates nearest neighbor connectivity).
            stripped_cz_rads: The angle in radians of the stripped cz gate.
            kwargs: Other desired `cq_compile` options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled `cirq.Circuit`(s).

        Raises:
            ValueError: If `target` is not a valid IBMQ target.
        """
        target = self._resolve_target(target)
        if not target.startswith("cq_"):
            raise ValueError(f"{target!r} is not a valid CQ target.")

        return self.compile(
            circuits,
            grid_shape=grid_shape,
            control_radius=control_radius,
            stripped_cz_rads=stripped_cz_rads,
            target=target,
            **kwargs,
        )

    def ibmq_compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str = "ibmq_qasm_simulator",
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) to the target IBMQ device.

        Qiskit Terra must be installed to correctly deserialize pulse schedules for pulse-enabled
        targets.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target IBMQ device.
            kwargs: Other desired `ibmq_compile` options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled `cirq.Circuit`(s), and whose
            .pulse_gate_circuit(s) attribute contains the corresponding pulse schedule(s) (when
            available).

        Raises:
            ValueError: If `target` is not a valid IBMQ target.
        """
        target = self._resolve_target(target)
        if not target.startswith("ibmq_"):
            raise ValueError(f"{target!r} is not a valid IBMQ target.")

        return self.compile(circuits, target=target, **kwargs)

    def compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the target device's native gateset.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target device.
            kwargs: Other desired compile options.

        Returns:
            A `CompilerOutput` object whose .circuit(s) attribute contains optimized compiled
            circuit(s).
        """

        target = self._resolve_target(target)

        if target.startswith("aqt_"):
            return self.aqt_compile(circuits, **kwargs)
        elif target.startswith("sandia_"):
            return self.qscout_compile(circuits, **kwargs)

        request_json = self._get_compile_request_json(circuits, target, **kwargs)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)
        json_dict = self._client.compile(request_json)
        return css.compiler_output.read_json(json_dict, circuits_is_list)

    def _get_compile_request_json(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Helper method to compile json dictionary."""

        css.validation.validate_cirq_circuits(circuits)
        serialized_circuits = css.serialization.serialize_circuits(circuits)
        request_json = {
            "cirq_circuits": serialized_circuits,
            "target": target,
        }
        options = {**self._client.client_kwargs, **kwargs}
        if options:
            request_json["options"] = cirq.to_json(options)
        return request_json

    def supercheq(
        self, files: List[List[int]], num_qubits: int, depth: int
    ) -> Tuple[List[cirq.Circuit], npt.NDArray[np.float_]]:
        """Returns the randomly generated circuits and the fidelity matrix for inputted files.

        Args:
            files: Input files from which to generate random circuits and fidelity matrix.
            num_qubits: The number of qubits to use to generate random circuits.
            depth: The depth of the random circuits to generate.

        Returns:
            A tuple containing the generated circuits and the fidelities for distinguishing files.
        """

        json_dict = self._client.supercheq(files, num_qubits, depth, "cirq_circuits")
        circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
        fidelities = gss.serialization.deserialize(json_dict["fidelities"])
        return circuits, fidelities

    def submit_dfe(
        self,
        rho_1: Tuple[cirq.AbstractCircuit, str],
        rho_2: Tuple[cirq.AbstractCircuit, str],
        num_random_bases: int,
        shots: int,
        **kwargs: Any,
    ) -> List[str]:
        """Executes the circuits neccessary for the DFE protocol.

        The circuits used to prepare the desired states should not contain final measurements, but
        can contain mid-circuit measurements (as long as the intended target supports them). For
        example, to prepare a Bell state to be ran in `ss_unconstrained_simulator`, you should pass
        `cirq.Circuit(cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1]))` as the first element of
        some `rho_i` (note there are no final measurements).

        The fidelity between states is calculated following the random measurement protocol
        outlined in [1].

        References:
            [1] Elben, Andreas, Benoît Vermersch, Rick van Bijnen, Christian Kokail, Tiff Brydges,
                Christine Maier, Manoj K. Joshi, Rainer Blatt, Christian F. Roos, and Peter Zoller.
                "Cross-platform verification of intermediate scale quantum devices." Physical
                review letters 124, no. 1 (2020): 010504.

        Args:
            rho_1: Tuple containing the information to prepare the first state. It contains a
                `cirq.Circuit` at index 0 and a target name at index 1.
            rho_2: Tuple containing the information to prepare the second state. It contains a
                `cirq.Circuit` at index 0 and a target name at index 1.
            num_random_bases: Number of random bases to measure each state in.
            shots: Number of shots to use per random basis.
            kwargs: Other execution parameters.
                - tag: Tag for all jobs submitted for this protocol.
                - lifespan: How long to store the jobs submitted for in days (only works with right
                permissions).
                - method: Which type of method to execute the circuits with.

        Returns:
            A list with two strings, which are the job ids that need to be passed to `process_dfe`
            to post-process the measurement results and return the fidelity.

        Raises:
            ValueError: If `circuit` is not a valid `cirq.Circuit`.
            SuperstaqServerException: If there was an error accessing the API.
        """
        circuit_1 = rho_1[0]
        circuit_2 = rho_2[0]
        target_1 = self._resolve_target(rho_1[1])
        target_2 = self._resolve_target(rho_2[1])

        css.validation.validate_cirq_circuits(circuit_1)
        css.validation.validate_cirq_circuits(circuit_2)

        if not (isinstance(circuit_1, cirq.Circuit) and isinstance(circuit_2, cirq.Circuit)):
            raise ValueError("Each state `rho_i` should contain a single circuit.")

        serialized_circuits_1 = css.serialization.serialize_circuits(circuit_1)
        serialized_circuits_2 = css.serialization.serialize_circuits(circuit_2)

        ids = self._client.submit_dfe(
            circuit_1={"cirq_circuits": serialized_circuits_1},
            target_1=target_1,
            circuit_2={"cirq_circuits": serialized_circuits_2},
            target_2=target_2,
            num_random_bases=num_random_bases,
            shots=shots,
            **kwargs,
        )

        return ids

    def process_dfe(self, ids: List[str]) -> float:
        """Process the results of a DFE protocol.

        Args:
            ids: A list (size two) of ids returned by a call to `submit_dfe`.

        Returns:
            The estimated fidelity between the two states as a float.

        Raises:
            ValueError: If `ids` is not of size two.
            SuperstaqServerException: If there was an error accesing the API or the jobs submitted
                through `submit_dfe` have not finished running.
        """
        return self._client.process_dfe(ids)

    def submit_aces(
        self,
        target: str,
        qubits: Sequence[int],
        shots: int,
        num_circuits: int,
        mirror_depth: int,
        extra_depth: int,
        method: Optional[str] = None,
        noise: Optional[Union[str, cirq.NoiseModel]] = None,
        error_prob: Optional[Union[float, Tuple[float, float, float]]] = None,
        tag: Optional[str] = None,
        lifespan: Optional[int] = None,
    ) -> str:
        """Submits the jobs to characterize `target` through the ACES protocol.

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
            target: The device target to characterize.
            qubits: A list with the qubit indices to characterize.
            shots: How many shots to use per circuit submitted.
            num_circuits: How many random circuits to use in the protocol.
            mirror_depth: The half-depth of the mirror portion of the random circuits.
            extra_depth: The depth of the fully random portion of the random circuits.
            method: Which type of method to execute the circuits with.
            noise: Noise model to simulate the protocol with. It can be either a string or a
                `cirq.NoiseModel`. Valid strings are "symmetric_depolarize", "phase_flip",
                "bit_flip" and "asymmetric_depolarize".
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
        if isinstance(noise, str):
            noise_dict["type"] = noise
            noise_dict["params"] = (
                (error_prob,) if isinstance(error_prob, numbers.Number) else error_prob
            )
        elif isinstance(noise, cirq.NoiseModel):
            noise_dict["cirq_noise_model"] = cirq.to_json(noise)

        return self._client.submit_aces(
            target=target,
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

    def target_info(self, target: str) -> Dict[str, Any]:
        """Returns information about device specified by `target`.

        Args:
            target: A string corresponding to a device.

        Returns:
            The corresponding device information.
        """
        target = self._resolve_target(target)
        return self._client.target_info(target)["target_info"]
