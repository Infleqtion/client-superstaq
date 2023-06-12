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
"""Service to access SuperstaQs API."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cirq
import general_superstaq as gss
import numpy as np
import numpy.typing as npt
from general_superstaq import ResourceEstimate, finance, logistics, superstaq_client, user_config

import cirq_superstaq as css


def _to_matrix_gate(matrix: npt.ArrayLike) -> cirq.MatrixGate:
    """Convert a unitary matrix into a cirq.MatrixGate acting either on qubits or on qutrits.

    Args:
        matrix: The (unitary) matrix to be converted.

    Returns:
        A cirq.MatrixGate with the given unitary.

    Raises:
        ValueError: If `matrix` could not be interpreted as a unitary gate acting on either qubits
            or qutrits.
    """

    matrix = np.asarray(matrix, dtype=complex)

    for dimension in (2, 3):
        num_qids = int(round(np.log(matrix.size) / np.log(dimension**2)))
        if matrix.shape == (dimension**num_qids, dimension**num_qids):
            qid_shape = (dimension,) * num_qids
            return cirq.MatrixGate(matrix, qid_shape=qid_shape)

    raise ValueError(
        "Could not determine qid_shape from array shape, consider using a cirq.MatrixGate instead."
    )


def counts_to_results(
    counter: Dict[str, int], circuit: cirq.AbstractCircuit, param_resolver: cirq.ParamResolver
) -> cirq.ResultDict:
    """Converts a collections.Counter to a cirq.ResultDict.

    Args:
            counter: The collections.Counter of counts for the run.
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
    for key in counter.keys():
        keys_as_list: List[int] = []

        # Combines the keys of the counter into a list. If key = "01", keys_as_list = [0, 1]
        for index in key:
            keys_as_list.append(int(index))

        # Gets the number of counts of the key
        # counter = collections.Counter({"01": 48, "11": 52})["01"] -> 48
        counts_of_key = counter[key]

        # Appends all the keys onto 'samples' list number-of-counts-in-the-key times
        # If collections.Counter({"01": 48, "11": 52}), [0, 1] is appended to 'samples` 48 times and
        # [1, 1] is appended to 'samples' 52 times
        for _ in range(counts_of_key):
            samples.append(keys_as_list)

    result = cirq.ResultDict(
        params=param_resolver,
        measurements={
            combine_key_names: np.array(samples),
        },
    )

    return result


def _validate_cirq_circuits(circuits: object) -> None:
    """Validates that the input is either a single `cirq.Circuit` or a list of `cirq.Circuit`
    instances.

    Args:
        circuits: The circuit(s) to run.

    Raises:
        ValueError: If the input is not a `cirq.Circuit` or a list of `cirq.Circuit` instances.
    """

    if not (
        isinstance(circuits, cirq.Circuit)
        or (
            isinstance(circuits, Sequence)
            and all(isinstance(circuit, cirq.Circuit) for circuit in circuits)
        )
    ):
        raise ValueError(
            "Invalid 'circuits' input. Must be a `cirq.Circuit` or a "
            "sequence of `cirq.Circuit` instances."
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


class Service(finance.Finance, logistics.Logistics, user_config.UserConfig):
    """A class to access SuperstaQ's API.

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
    ) -> None:
        """Creates the Service to access SuperstaQ's API.

        Args:
            api_key: A string that allows access to the SuperstaQ API. If no key is provided, then
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

        Raises:
            EnvironmentError: If an API key was not provided and could not be found.
        """
        self.default_target = default_target

        self._client = superstaq_client._SuperstaQClient(
            client_name="cirq-superstaq",
            remote_host=remote_host,
            api_key=api_key,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def _resolve_target(self, target: Union[str, None]) -> str:
        target = target or self.default_target
        if not target:
            raise ValueError(
                "This call requires a target, but none was provided and default_target is not set."
            )

        return target

    def get_counts(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Runs the given circuit on the SuperstaQ API and returns the result
        of the ran circuit as a collections.Counter

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            target: Where to run the job.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.
            method: Optional execution method.
            options: Optional dictionary of optimization and execution parameters

        Returns:
            A `collection.Counter` for running the circuit.
        """
        _validate_cirq_circuits(circuit)
        _validate_integer_param(repetitions)
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
        job = self.create_job(resolved_circuit, int(repetitions), target, method, options)
        counts = job.counts()

        return counts

    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> cirq.ResultDict:
        """Run the given circuit on the SuperstaQ API and returns the result
        of the ran circut as a cirq.ResultDict.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            target: Where to run the job.
            method: Execution method.
            options: Optional dictionary of optimization and execution parameters
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `cirq.ResultDict` for running the circuit.
        """
        counts = self.get_counts(circuit, repetitions, target, param_resolver, method, options)
        return counts_to_results(counts, circuit, param_resolver)

    def sampler(self, target: Optional[str] = None) -> cirq.Sampler:
        """Returns a `cirq.Sampler` object for accessing sampler interface.

        Args:
            target: Target to sample against.

        Returns:
            A `cirq.Sampler` for the SuperstaQ API.
        """
        target = self._resolve_target(target)
        return css.sampler.Sampler(service=self, target=target)

    def create_job(
        self,
        circuit: cirq.AbstractCircuit,
        repetitions: int = 1000,
        target: Optional[str] = None,
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> css.job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 1000.
            target: Where to run the job.
            method: Execution method.
            options: Optional dictionary of optimization and execution parameters

        Returns:
            A `css.Job` which can be queried for status or results.

        Raises:
            ValueError: If the circuit has no measurements to sample.
            SuperstaQException: If there was an error accessing the API.
        """
        if not circuit.has_measurements():
            # TODO: only raise if the run method actually requires samples (and not for e.g. a
            # statevector simulation)
            raise ValueError("Circuit has no measurements to sample.")

        serialized_circuits = css.serialization.serialize_circuits(circuit)

        target = self._resolve_target(target)

        result = self._client.create_job(
            serialized_circuits={"cirq_circuits": serialized_circuits},
            repetitions=repetitions,
            target=target,
            method=method,
            options=options,
        )
        # The returned job does not have fully populated fields; they will be filled out by
        # when the new job's status is first queried
        return self.get_job(result["job_ids"][0])

    def get_job(self, job_id: str) -> css.job.Job:
        """Gets a job that has been created on the SuperstaQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `css.Job` which can be queried for status or results.

        Raises:
            SuperstaQNotFoundException: If there was no job with the given `job_id`.
            SuperstaQException: If there was an error accessing the API.
        """
        return css.job.Job(client=self._client, job_id=job_id)

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
        """Get the querying user's account balance in USD.

        Args:
            pretty_output: Whether to return a pretty string or a float of the balance.

        Returns:
            If pretty_output is True, returns the balance as a nicely formatted string ($-prefix,
                commas on LHS every three digits, and two digits after period). Otherwise, simply
                returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"${balance:,.2f}"
        return balance

    def get_targets(self) -> Dict[str, List[str]]:
        """Get list of available targets."""
        return self._client.get_targets()["superstaq_targets"]

    def resource_estimate(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]], target: Optional[str] = None
    ) -> Union[ResourceEstimate, List[ResourceEstimate]]:
        """Generates resource estimates for circuit(s).

        Args:
            circuits: Cirq Circuit(s).
            target: String of target representing target device
        Returns:
            ResourceEstimate(s) containing resource costs (after compilation)
        """
        circuit_is_list = isinstance(circuits, List)
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

    def aqt_compile(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        target: str = "aqt_keysight_qpu",
        atol: Optional[float] = None,
        gate_defs: Optional[
            Mapping[str, Union[npt.NDArray[np.complex_], cirq.Gate, cirq.Operation, None]]
        ] = None,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) for the Advanced Quantum Testbed (AQT) at
        Lawrence Berkeley National Laboratory.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target AQT device.
            atol: An optional tolerance to use for approximate gate synthesis.
            gate_defs: An optional dictionary mapping names in qtrl configs to operations, where
                each operation can be a unitary matrix, `cirq.Gate`, `cirq.Operation`, or None. More
                specific associations take precedence, for example `{"SWAP": cirq.SQRT_ISWAP,
                "SWAP/C5C4": cirq.SQRT_ISWAP_INV}` implies `SQRT_ISWAP` for all "SWAP" calibrations
                except "SWAP/C5C4" (which will instead be mapped to a `SQRT_ISWAP_INV` gate on
                qubits 4 and 5). Setting any calibration to None will disable that calibration.
            kwargs: Other desired aqt_compile options.

        Returns:
            Object whose .circuit(s) attribute contains the optimized circuits(s). If qtrl is
            installed, the object's .seq attribute is a qtrl Sequence object containing pulse
            sequences for each compiled circuit, and its .pulse_list(s) attribute contains the
            corresponding list(s) of cycles.
        """

        _validate_cirq_circuits(circuits)
        return self._aqt_compile(circuits, target=target, atol=atol, gate_defs=gate_defs, **kwargs)

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
            gate_defs: An optional dictionary mapping names in qtrl configs to operations, where
                each operation can be a unitary matrix, `cirq.Gate`, `cirq.Operation`, or None. More
                specific associations take precedence, for example `{"SWAP": cirq.SQRT_ISWAP,
                "SWAP/C5C4": cirq.SQRT_ISWAP_INV}` implies `SQRT_ISWAP` for all "SWAP" calibrations
                except "SWAP/C5C4" (which will instead be mapped to a `SQRT_ISWAP_INV` gate on
                qubits 4 and 5). Setting any calibration to None will disable that calibration.
            kwargs: Other desired aqt_compile_eca options.

        Returns:
            Object whose .circuits attribute is a list (or list of lists) of logically equivalent
            circuits. If qtrl is installed, the object's .seq attribute is a qtrl Sequence object
            containing pulse sequences for each compiled circuit, and its .pulse_list(s) attribute
            contains the corresponding list(s) of cycles.
        """
        _validate_cirq_circuits(circuits)
        _validate_integer_param(num_equivalent_circuits)

        return self._aqt_compile(
            circuits,
            target=target,
            num_equivalent_circuits=num_equivalent_circuits,
            random_seed=random_seed,
            atol=atol,
            gate_defs=gate_defs,
            **kwargs,
        )

    def _aqt_compile(
        self,
        circuits: Union[cirq.Circuit, Sequence[cirq.Circuit]],
        target: str,
        *,
        num_equivalent_circuits: Optional[int] = None,
        random_seed: Optional[int] = None,
        atol: Optional[float] = None,
        gate_defs: Optional[
            Mapping[str, Union[npt.NDArray[np.complex_], cirq.Gate, cirq.Operation, None]]
        ] = None,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Generic API for compiling circuits for AQT devices. See `Service.aqt_compile()` and
        `Service.aqt_compile_eca()`.
        """
        _validate_cirq_circuits(circuits)

        serialized_circuits = css.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)

        request_json = {
            "cirq_circuits": serialized_circuits,
            "target": target,
        }

        options_dict: Dict[str, Union[float, Dict[str, Union[cirq.Gate, cirq.Operation, None]]]]
        options_dict = {**kwargs}

        if num_equivalent_circuits is not None:
            _validate_integer_param(num_equivalent_circuits)
            options_dict["num_eca_circuits"] = num_equivalent_circuits
        if random_seed is not None:
            options_dict["random_seed"] = random_seed
        if atol is not None:
            options_dict["atol"] = atol
        if gate_defs is not None:
            gate_defs_cirq = {}
            for key, val in gate_defs.items():
                if val is not None and not isinstance(val, (cirq.Gate, cirq.Operation)):
                    val = _to_matrix_gate(val).with_name(key)
                gate_defs_cirq[key] = val
            options_dict["gate_defs"] = gate_defs_cirq
        if options_dict:
            request_json["options"] = cirq.to_json(options_dict)

        json_dict = self._client.post_request("/aqt_compile", request_json)
        return css.compiler_output.read_json_aqt(
            json_dict, circuits_is_list, num_equivalent_circuits
        )

    def qscout_compile(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        mirror_swaps: bool = False,
        base_entangling_gate: str = "xx",
        target: str = "sandia_qscout_qpu",
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
            target: String of target representing target device
            mirror_swaps: Whether to use mirror swapping to reduce two-qubit gate overhead.
            base_entangling_gate: The base entangling gate to use (either "xx" or "zz").
            kwargs: Other desired qscout_compile options.

        Returns:
            Object whose .circuit(s) attribute contains optimized `cirq.Circuit`(s), and
            `.jaqal_program(s)` attribute contains the corresponding Jaqal program(s).

        Raises:
            ValueError: If `base_entangling_gate` is not a valid gate option.
        """
        _validate_cirq_circuits(circuits)

        if base_entangling_gate not in ("xx", "zz"):
            raise ValueError("base_entangling_gate must be either 'xx' or 'zz'")

        serialized_circuits = css.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)

        options_dict = {
            "mirror_swaps": mirror_swaps,
            "base_entangling_gate": base_entangling_gate,
            **kwargs,
        }

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
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        target: str = "cq_hilbert_qpu",
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) to the target CQ device.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target CQ device.
            kwargs: Other desired cq_compile options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled `cirq.Circuit`(s).
        """
        return self.compile(circuits, target=target, **kwargs)

    def ibmq_compile(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        target: str = "ibmq_qasm_simulator",
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles and optimizes the given circuit(s) to the target IBMQ device.

        Qiskit Terra must be installed to correctly deserialize pulse schedules for pulse-enabled
        targets.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target IBMQ device.
            kwargs: Other desired ibmq_compile options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled `cirq.Circuit`(s), and whose
            .pulse_sequence(s) attribute contains the corresponding pulse schedule(s) (when
            available).

        Raises:
            ValueError: If `target` is not a valid IBMQ target.
        """
        target = self._resolve_target(target)
        if not target.startswith("ibmq_"):
            raise ValueError(f"{target} is not an IBMQ target")

        return self.compile(circuits, target=target, **kwargs)

    def compile(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        target: str,
        **kwargs: Any,
    ) -> css.compiler_output.CompilerOutput:
        """Compiles the given circuit(s) to the target device.

        Args:
            circuits: The circuit(s) to compile.
            target: String of target device.
            kwargs: Other desired compilation options.

        Returns:
            Object whose .circuit(s) attribute contains the compiled `cirq.Circuit`(s).
        """
        _validate_cirq_circuits(circuits)
        serialized_circuits = css.serialization.serialize_circuits(circuits)
        circuits_is_list = not isinstance(circuits, cirq.Circuit)

        target = self._resolve_target(target)

        request_json = {
            "cirq_circuits": serialized_circuits,
            "target": target,
            "options": cirq.to_json(kwargs),
        }

        json_dict = self._client.compile(request_json)

        return css.compiler_output.read_json(json_dict, circuits_is_list)

    def supercheq(
        self, files: List[List[int]], num_qubits: int, depth: int
    ) -> Tuple[List[cirq.Circuit], npt.NDArray[np.float_]]:
        """Returns the randomly generated circuits and the fidelity matrix for inputted files."""

        _validate_integer_param(num_qubits)
        _validate_integer_param(depth)
        json_dict = self._client.supercheq(files, num_qubits, depth, "cirq_circuits")
        circuits = css.serialization.deserialize_circuits(json_dict["cirq_circuits"])
        fidelities = gss.serialization.deserialize(json_dict["fidelities"])
        return circuits, fidelities

    def target_info(self, target: str) -> Dict[str, Any]:
        """Returns information about device specified by `target`."""
        return self._client.target_info(target)["target_info"]
