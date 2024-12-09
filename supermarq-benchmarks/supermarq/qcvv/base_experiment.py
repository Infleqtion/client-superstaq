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
"""Base experiment class and tools used across all experiments.
"""
from __future__ import annotations

import functools
import math
import pprint
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class Sample:
    """A sample circuit to use along with any data about the circuit
    that is needed for analysis
    """

    raw_circuit: cirq.Circuit
    """The raw (i.e. pre-compiled) sample circuit."""
    data: dict[str, Any]
    """The corresponding data about the circuit"""
    probabilities: dict[str, float] | None = None
    """The probabilities of the computational basis states"""
    job: css.Job | None = None
    """The superstaq job corresponding to the sample. Defaults to None if no job is
    associated with the sample."""
    compiled_circuit: cirq.Circuit | None = None
    """The compiled circuit. Only used if the circuits are compiled for a specific
    target."""

    @property
    def target(self) -> str:
        """Returns:
        The name of the target that the sample was submitted to.
        """
        if self.job is not None:
            # If there is a job then get the target
            return self.job.target()

        if self.probabilities is not None:
            # If no job, but probabilities have been calculated, infer that a local
            # simulator was used.
            return "Local simulator"

        # Otherwise the experiment hasn't yet been run so there is no target.
        return "No target"

    @property
    def circuit(self) -> cirq.Circuit:
        """Returns:
        The circuit used for the experiment. Defaults to the compiled circuit if available
        and if not returns the raw circuit.
        """
        if self.compiled_circuit is not None:
            return self.compiled_circuit

        return self.raw_circuit


@dataclass(frozen=True)
class BenchmarkingResults(ABC):
    """A dataclass for storing the results of the experiment. Requires subclassing for
    each new experiment type."""

    target: str
    """The target device that was used."""
    total_circuits: int
    """The total number of circuits used in the experiment."""

    experiment_name: str = field(init=False)
    """The name of the experiment."""


ResultsT = TypeVar("ResultsT", bound="BenchmarkingResults")


class BenchmarkingExperiment(ABC, Generic[ResultsT]):
    """Base class for gate benchmarking experiments.

    The interface for implementing these experiments is as follows:

    #. First instantiate the desired experiment object

        .. code::

            experiment = ExampleExperiment(<<args/kwargs>>)

    #. Prepare the circuits and run the experiment on the desired target. This can either be a
       custom simulator or a real device name. For example:

        .. code::

            noise_model = cirq.depolarize(p=0.01, n_qubits=1)
            sim = cirq.DensityMatrixSimulator(noise=noise_model)

            experiment.prepare_experiment(<<args/kwargs>>)
            experiment.run_with_simulator(simulator=sim, <<args/kwargs>>)

    #. Then we analyse the results. If the target was a local simulator this will be available as
       soon as the :code:`run_with_simulator()` method has finished executing. On the other hand
       if a real device was accessed via Superstaq then it may take time for the data to be
       available from the server. The :code:`collect_data()` will return :code:`True` when all
       data has been collected and is ready to be analysed.

       .. code::

            if experiment.collect_data():
                results = experiment.analyze_results(<<args>>)

    #. The final results of the experiment will be stored in the :code:`results` attribute as a
       :class:`BenchmarkingResults` of values, while all the data from the experiment will be
       stored in the :code:`raw_data` attribute as a :class:`~pandas.DataFrame`. Some experiments
       may include additional data attributes for data generated during the analysis.

        .. code::

            results = experiment.results
            data = experiment.raw_data

    Additionally it is possible to pre-compile the experimental circuits for a given device using

    .. code::

        experiment.prepare_experiment(<<args/kwargs>>)
        experiment.compile_circuits(target=<<target_name>>)

    And then to run the experiment using a custom callable function for evaluating the circuits.
    For example this could be a function that uses a connection to a local device.

    .. code::

        experiment.run_with_callable(<<function_name>>)

    When implementing a new experiment, 4 methods need to be implemented:

    #. :meth:`_build_circuits`: Given a number of circuits and an iterable of the different numbers
        of layers to use, return a list of :class:`Sample` objects that need to be sampled during
        the experiment.

    #. :meth:`_process_probabilities`: Take the probability distribution over the
        computational basis resulting from running each circuit and combine the relevant details
        into a :class:`pandas.DataFrame`.

    #. :meth:`analyze_results`: Analyse the data in the :attr:`raw_data` dataframe and return a
        :class:`BenchmarkingResults` object containing the results of the experiment.

    #. :meth:`plot_results`: Produce any relevant plots that are useful for understanding the
        results of the experiment.

    Additionally the :class:`BenchmarkingResults` dataclass needs subclassing to hold the specific
    results of the new experiment.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        random_seed: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a benchmarking experiment.

        Args:
            num_qubits: The number of qubits used during the experiment. Most subclasses
                will determine this from their other inputs.
            random_seed: An optional seed to use for randomization.
            kwargs: Additional kwargs passed to the Superstaq service object.
        """
        self.qubits = cirq.LineQubit.range(num_qubits)
        """The qubits used in the experiment."""

        self._raw_data: pd.DataFrame | None = None
        "The data generated during the experiment"

        self._results: ResultsT | None = None
        """The attribute to store the results in."""

        self._samples: Sequence[Sample] | None = None
        """The attribute to store the experimental samples in."""

        self._service_kwargs = kwargs
        """Arguments to pass to the Superstaq service for submitting jobs."""

        self._rng = np.random.default_rng(random_seed)

    ##############
    # Properties #
    ##############
    @functools.cached_property
    def _superstaq_service(self) -> css.Service:
        """A Superstaq service to use for compilation and circuit submission."""
        return css.Service(**self._service_kwargs)

    @property
    def num_qubits(self) -> int:
        """Returns:
        The number of qubits used in the experiment
        """
        return len(self.qubits)

    @property
    def raw_data(self) -> pd.DataFrame:
        """The data from the most recently run experiment.

        Raises:
            RuntimeError: If no results are available.
        """
        if self._raw_data is None:
            raise RuntimeError("No data to retrieve. The experiment has not been run.")

        return self._raw_data

    @property
    def results(self) -> ResultsT:
        """The results from the most recently run experiment.

        Raises:
            RuntimeError: If no results are available.
        """
        if self._results is None:
            raise RuntimeError("No results to retrieve. The experiment has not been run.")

        return self._results

    @property
    def samples(self) -> Sequence[Sample]:
        """The samples generated during the experiment.

        Raises:
            RuntimeError: If no samples are available.
        """
        if self._samples is None:
            raise RuntimeError("No samples to retrieve. The experiment has not been run.")

        return self._samples

    @property
    def targets(self) -> frozenset[str]:
        """Returns:
        A set of the unique target that each sample was submitted to
        """
        return frozenset(sample.target for sample in self.samples)

    ###################
    # Private Methods #
    ###################
    @abstractmethod
    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
    ) -> Sequence[Sample]:
        """Build a list of circuits required for the experiment. These circuits are stored in
        :class:`Sample` objects along with any additional data that is needed during the analysis.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different cycle depths to use during the experiment.

        Returns:
           The list of experiment samples.
        """

    @staticmethod
    def _interleave_op(
        circuit: cirq.Circuit, operation: cirq.Operation, include_final: bool = False
    ) -> cirq.Circuit:
        """Interleave a given operation into a circuit.

        Args:
            circuit: The original circuit.
            operation: The operation to interleave.
            include_final: If True then the interleaving gate is also appended to
                the end of the circuit.

        Returns:
            A copy of the original circuit with the provided gate interleaved.
        """
        operation = operation.with_tags("no_compile")
        interleaved_circuit = circuit.copy()
        interleaved_circuit.batch_insert(
            [(k, operation) for k in range(len(circuit) - int(not include_final), 0, -1)]
        )
        return interleaved_circuit

    def _process_device_counts(self, counts: dict[str, int]) -> dict[str, float]:
        """Process the counts returned by the server into a dictionary of probabilities.

        Args:
            counts: A dictionary of the observed counts for each state in the computational basis.

        Returns:
            A dictionary of the probability of each state in the computational basis.
        """
        total = sum(counts.values())

        probabilities = {
            format(idx, f"0{self.num_qubits}b"): 0.0 for idx in range(2**self.num_qubits)
        }

        for key, count in counts.items():
            probabilities[key] = count / total

        return probabilities

    @abstractmethod
    def _process_probabilities(self, samples: Sequence[Sample]) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into a data frame
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """

    def _retrieve_jobs(self) -> dict[str, str]:
        """Retrieve the jobs from the server.

        Returns:
            A dictionary of the statuses of any jobs that have not been successfully completed.
        """
        # If no jobs then return empty dictionary
        if not [sample for sample in self.samples if sample.job is not None]:
            return {}

        statuses = {}
        waiting_samples = [sample for sample in self.samples if sample.probabilities is None]
        for sample in tqdm(waiting_samples, "Retrieving jobs"):
            if sample.job is None:
                continue
            if (
                job_status := sample.job.status()
            ) in css.job.Job.NON_TERMINAL_STATES + css.job.Job.UNSUCCESSFUL_STATES:
                statuses[sample.job.job_id()] = job_status
            else:
                sample.probabilities = self._process_device_counts(sample.job.counts(0))

        return statuses

    def _has_raw_data(self) -> None:
        """Checks if any of the samples already have probabilities stored. If so raises a runtime
        error to prevent them from being overwritten.

        To be used within all `run` methods to prevent data being overwritten

        Raises:
            RuntimeError: If any samples already have probabilities stored.
        """
        if any(sample.probabilities is not None for sample in self.samples):
            raise RuntimeError(
                "Some samples have already been run. Re-running the experiment will"
                "overwrite these results. If this is the desired behaviour use `overwrite=True`"
            )

    def _sample_statuses(self) -> list[str | None]:
        """Returns:
        The statuses of the jobs associated with each sample. If no job is associated
        with a sample then :code:`None` is listed instead.
        """
        statuses: list[str | None] = []
        for sample in self.samples:
            if sample.job is None:
                statuses.append(None)
            else:
                statuses.append(sample.job.status())
        return statuses

    def _validate_circuits(self) -> None:
        """Checks that all circuits contain a terminal measurement of all qubits."""
        for sample in self.samples:
            if not sample.circuit.has_measurements():
                raise ValueError("QCVV experiment circuits must contain measurements.")
            if not sample.circuit.are_all_measurements_terminal():
                raise ValueError("QCVV experiment circuits can only contain terminal measurements.")
            if not sorted(sample.circuit[-1].qubits) == sorted(self.qubits):
                raise ValueError(
                    "The terminal measurement in QCVV experiment circuits must measure all qubits."
                )

    ###################
    # Public Methods  #
    ###################
    @abstractmethod
    def analyze_results(self, plot_results: bool = True) -> ResultsT:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """

    def collect_data(self, force: bool = False) -> bool:
        """Collect the data from the samples and process it into the :attr:`raw_data` attribute.

        If the data is successfully stored in the :attr:`raw_data` attribute then the function will
        return :code:`True`.

        If either not all jobs submitted to the server have completed, or not all samples have
        probability results then no data will be saved in :attr:`raw_data` and the function will
        return :code:`False`. This check can be overridden with :code:`force=True` in which case
        only the samples which have probability results will be used to generate the results
        dataframe.

        Args:
            force: Whether to override the check that all data is present. Defaults to False.

        Raises:
            RuntimeError: If :code:`force=True` but there are no samples with any data.
            RuntimeError: If the experiment has not yet been run.

        Returns:
            Whether the results dataframe has been successfully created.
        """
        if not self._samples:
            raise RuntimeError("The experiment has not yet ben run.")

        # Retrieve jobs from server (if needed)
        outstanding_statuses = self._retrieve_jobs()
        if outstanding_statuses:
            print(
                "Not all circuits have been sampled. "
                "Please wait and try again.\n"
                f"Outstanding Superstaq jobs:\n{pprint.pformat(outstanding_statuses)}"
            )
            if not force:
                return False

        completed_samples = [sample for sample in self.samples if sample.probabilities is not None]

        if not len(completed_samples) == len(self.samples):
            print("Some samples do not have probability results.")
            if not force:
                return False

        if len(completed_samples) == 0:
            raise RuntimeError("Cannot force data collection when there are no completed samples.")

        self._raw_data = self._process_probabilities(completed_samples)
        return True

    def compile_circuits(self, target: str, **kwargs: Any) -> None:
        """Compiles the experiment circuits for the given device. Useful if the samples
        are not going to be run via Superstaq.

        Args:
            target: The device to compile to.
            kwargs: Additional desired compile options.
        """
        compiled_circuits = self._superstaq_service.compile(
            [sample.circuit for sample in self.samples], target=target, **kwargs
        ).circuits

        for k, sample in enumerate(self.samples):
            sample.compiled_circuit = compiled_circuits[k]  # type: ignore[assignment]

    @abstractmethod
    def plot_results(self) -> None:
        """Plot the results of the experiment"""

    def prepare_experiment(
        self, num_circuits: int, cycle_depths: Iterable[int], overwrite: bool = False
    ) -> None:
        """Prepares the circuits needed for the experiment

        Args:
            num_circuits: Number of circuits to run.
            cycle_depths: An iterable of the different layer depths to use during the experiment.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.

        Raises:
            RuntimeError: If the experiment has already been run once and the `overwrite` argument
                is not True
            ValueError: If any of the cycle depths provided negative or zero.
        """
        if self._samples is not None and not overwrite:
            raise RuntimeError(
                "This experiment already has existing data which would be overwritten by "
                "rerunning the experiment. If this is the desired behavior set `overwrite=True`"
            )

        if any(depth <= 0 for depth in cycle_depths):
            raise ValueError("The `cycle_depths` iterator can only include positive values.")

        self._samples = self._build_circuits(num_circuits, cycle_depths)
        self._validate_circuits()

    def run_on_device(
        self,
        target: str,
        repetitions: int = 10_000,
        method: str | None = None,
        overwrite: bool = False,
        **target_options: Any,
    ) -> css.Job:
        """Submit the circuit samples to the desired target device and store the resulting
        probabilities.

        The set of circuits is partitioned as necessary to not exceed the maximum circuits that can
        be submitted to the given target device. The function then waits for the jobs to complete
        before saving the resulting probability distributions.

        Args:
            target: The name of a Superstaq target.
            repetitions: The number of shots to sample. Defaults to 10,000.
            method: Optional method to use on the Superstaq device. Defaults to None corresponding
                to normal running.
            target_options: Optional configuration dictionary passed when submitting the job.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.

        Return:
            The superstaq job containing all the circuits submitted as part of the experiment.
        """
        if not overwrite:
            self._has_raw_data()

        experiment_job = self._superstaq_service.create_job(
            [sample.circuit for sample in self.samples],
            target=target,
            method=method,
            repetitions=repetitions,
            **target_options,
        )
        compiled_circuits = experiment_job.compiled_circuits()

        for k, sample in enumerate(self.samples):
            sample.job = experiment_job[k]
            sample.compiled_circuit = compiled_circuits[k]

        return experiment_job

    def run_with_simulator(
        self,
        simulator: cirq.Sampler | None = None,
        repetitions: int = 10_000,
        overwrite: bool = False,
    ) -> None:
        """Use the local simulator to sample the circuits and store the resulting probabilities.

        Args:
            simulator: A local :class:`~cirq.Sampler` to use. If None then the default
                :class:`cirq.Simulator` simulator is used. Defaults to None.
            repetitions: The number of shots to sample. Defaults to 10,000.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.
        """
        if not overwrite:
            self._has_raw_data()

        if simulator is None:
            simulator = cirq.Simulator(seed=self._rng)

        for sample in tqdm(self.samples, desc="Simulating circuits"):
            result = simulator.run(sample.circuit, repetitions=repetitions)
            hist = result.histogram(key=cirq.measurement_key_name(sample.circuit))
            sample.probabilities = {
                f"{i:0{self.num_qubits}b}": 0.0 for i in range(2**self.num_qubits)
            }  # Set all probabilities to zero
            for val, count in hist.items():
                # Add in results from the histogram
                sample.probabilities[f"{val:0{self.num_qubits}b}"] = count / repetitions

    def run_with_callable(
        self,
        circuit_eval_func: Callable[[cirq.Circuit], dict[str, float]],
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Evaluates the probabilities for each circuit using a user provided callable function.
        This function should take a circuit as input and return a dictionary of probabilities for
        each bitstring (including states with zero probability).

        Args:
            circuit_eval_func: The custom function to use when evaluating circuit probabilities.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.
            kwargs: Additional arguments to pass to the custom function.

        Raises:
            RuntimeError: If the returned probabilities dictionary keys is missing include
                an incorrect number of bits.
            RuntimeError: If the returned probabilities dictionary values do not sum to 1.0.
        """
        if not overwrite:
            self._has_raw_data()
        for sample in tqdm(self.samples, desc="Running circuits"):
            probability = circuit_eval_func(sample.circuit, **kwargs)
            if not all(len(key) == self.num_qubits for key in probability.keys()):
                raise RuntimeError("Returned probabilities include an incorrect number of bits.")
            if not math.isclose(sum(probability.values()), 1.0):
                raise RuntimeError("Returned probabilities do not sum to 1.0.")

            for k in range(2**self.num_qubits):
                if (bitstring := format(k, f"0{self.num_qubits}b")) not in probability:
                    probability[bitstring] = 0.0

            sample.probabilities = probability
