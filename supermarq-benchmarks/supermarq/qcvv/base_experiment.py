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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

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

    circuit: cirq.Circuit
    """The raw (i.e. pre-compiled) sample circuit."""
    data: dict[str, Any]
    """The corresponding data about the circuit that is needed when analyzing results
    (e.g. cycle depth)."""


# @dataclass(frozen=True)
@dataclass
class QCVVResults(ABC):
    """A dataclass for storing the data and analyze results of the experiment. Requires
    subclassing for each new experiment type."""

    target: str
    """The target device that was used."""

    experiment: QCVVExperiment
    """Reference to the underlying experiment that generated these results experiment."""

    job: css.Job | None = None
    """The associated Superstaq job (if applicable)."""

    data: pd.DataFrame | None = None
    """The raw data generated."""

    @property
    def data_ready(self) -> bool:
        if self.data is not None:
            return True
        if self.job is None:
            raise RuntimeError()  # TODO
        job_status = self.job.status()
        if job_status == "Done":
            self.data = self._collect_device_counts()
            return True
        return False

    @property
    def samples(self):
        return self.experiment.samples

    @property
    def num_qubits(self):
        return self.experiment.num_qubits

    @property
    def num_circuits(self):
        return self.experiment.num_circuits

    def analyze_results(self, plot_results: bool = True, print_results: bool = True) -> None:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """
        if not self.data_ready:
            warnings.warn(
                "Experiment data is not yet ready to analyse. This is likely because "
                "the Superstaq job has not yet been completed. Either wait and try again "
                "later, or interrogate the `.job` attribute."
            )
            return

        self._analyze_results()

        if plot_results:
            self.plot_results()

        if print_results:
            self.print_results()

    @abstractmethod
    def _analyze_results(self) -> None:
        """"""

    @abstractmethod
    def plot_results(self) -> None:
        """Plot the results of the experiment"""

    @abstractmethod
    def print_results(self) -> None:
        """Prints the key results data."""

    def _collect_device_counts(self) -> pd.DataFrame:
        """Process the counts returned by the server into a dictionary of probabilities.

        Args:
            counts: A dictionary of the observed counts for each state in the computational basis.

        Returns:
            A dictionary of the probability of each state in the computational basis.
        """
        records = []
        device_counts = self.job.counts()
        for counts, sample in zip(device_counts, self.samples):

            total = sum(counts.values())
            probabilities = {
                format(idx, f"0{self.num_qubits}b"): 0.0 for idx in range(2**self.num_qubits)
            }
            for key, count in counts.items():
                probabilities[key] = count / total
            records.append({**sample.data, **probabilities})

        return pd.DataFrame(records)


class QCVVExperiment(ABC):
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
        num_circuits: int,
        cycle_depths: Iterable[int],
        *,
        random_seed: int | np.random.Generator | None = None,
        results_cls: QCVVResults = QCVVResults,
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

        self.num_circuits = num_circuits
        """The number of circuits to build for each cycle depth."""

        self.cycle_depths = cycle_depths
        """The different cycle depths to test at."""

        self._raw_data: pd.DataFrame | None = None
        "The data generated during the experiment"

        self._samples: Sequence[Sample] | None = None
        """The attribute to store the experimental samples in."""

        self._service_kwargs = kwargs
        """Arguments to pass to the Superstaq service for submitting jobs."""

        self._rng = np.random.default_rng(random_seed)

        self._results_cls = results_cls

        self._prepare_experiment()
        """Create all the samples needed for the experiment."""

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
    def samples(self) -> Sequence[Sample]:
        """The samples generated during the experiment.

        Raises:
            RuntimeError: If no samples are available.
        """
        if self._samples is None:
            raise RuntimeError("No samples to retrieve. The experiment has not been run.")

        return self._samples

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

    def _prepare_experiment(
        self,
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

        if any(depth <= 0 for depth in self.cycle_depths):
            raise ValueError("The `cycle_depths` iterator can only include positive values.")

        self._samples = self._build_circuits(self.num_circuits, self.cycle_depths)
        self._validate_circuits()

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
    def run_on_device(
        self,
        target: str,
        repetitions: int = 10_000,
        method: str | None = None,
        **target_options: Any,
    ) -> QCVVResults:
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

        experiment_job = self._superstaq_service.create_job(
            [sample.circuit for sample in self.samples],
            target=target,
            method=method,
            repetitions=repetitions,
            **target_options,
        )

        return self._results_cls(
            target=target,
            experiment=self,
            job=experiment_job,
        )

    def run_with_simulator(
        self,
        simulator: cirq.Sampler | None = None,
        repetitions: int = 10_000,
    ) -> QCVVResults:
        """Use the local simulator to sample the circuits and store the resulting probabilities.

        Args:
            simulator: A local :class:`~cirq.Sampler` to use. If None then the default
                :class:`cirq.Simulator` simulator is used. Defaults to None.
            repetitions: The number of shots to sample. Defaults to 10,000.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.
        """
        if simulator is None:
            simulator = cirq.Simulator(seed=self._rng)

        records = []
        for sample in tqdm(self.samples, desc="Simulating circuits"):
            result = simulator.run(sample.circuit, repetitions=repetitions)
            hist = result.histogram(key=cirq.measurement_key_name(sample.circuit))
            probabilities = {
                f"{i:0{self.num_qubits}b}": 0.0 for i in range(2**self.num_qubits)
            }  # Set all probabilities to zero
            for val, count in hist.items():
                # Add in results from the histogram
                probabilities[f"{val:0{self.num_qubits}b}"] = count / repetitions
            records.append({**sample.data, **probabilities})

        return self._results_cls(
            target="LocalSimulator",
            experiment=self,
            data=pd.DataFrame(records),
        )

    def run_with_callable(
        self,
        circuit_eval_func: Callable[[cirq.Circuit], dict[str | int, float]],
        **kwargs: Any,
    ) -> QCVVResults:
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
        records = []
        for sample in tqdm(self.samples, desc="Running circuits"):
            probability = circuit_eval_func(sample.circuit, **kwargs)
            if not all(len(key) == self.num_qubits for key in probability.keys()):
                raise RuntimeError("Returned probabilities include an incorrect number of bits.")
            if not math.isclose(sum(probability.values()), 1.0):
                raise RuntimeError("Returned probabilities do not sum to 1.0.")

            for k in range(2**self.num_qubits):
                if (bitstring := format(k, f"0{self.num_qubits}b")) not in probability:
                    probability[bitstring] = 0.0

            records.append({**sample.data, **probability})

        return self._results_cls(
            target="Callable",
            experiment=self,
            data=pd.DataFrame(records),
        )
