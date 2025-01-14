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

    circuit_index: int
    """The index of the circuit. There will be D samples with matching circuit index, one for each
    cycle depth being measured. This index is useful for grouping results during analysis.
    """
    circuit: cirq.Circuit
    """The raw (i.e. pre-compiled) sample circuit."""
    data: dict[str, Any]
    """The corresponding data about the circuit that is needed when analyzing results
    (e.g. cycle depth)."""


@dataclass
class QCVVResults(ABC):
    """A dataclass for storing the data and analyze results of the experiment. Requires
    subclassing for each new experiment type."""

    target: str
    """The target device that was used."""

    experiment: QCVVExperiment[QCVVResults]
    """Reference to the underlying experiment that generated these results experiment."""

    job: css.Job | None = None
    """The associated Superstaq job (if applicable)."""

    data: pd.DataFrame | None = None
    """The raw data generated."""

    @property
    def data_ready(self) -> bool:
        """Whether the experimental data is ready to analyse.

        Raises:
            RuntimeError: If their is no stored data and no Superstaq job to use to collect the
                results.
        """
        if self.data is not None:
            return True
        if self.job is None:
            raise RuntimeError(
                "No data available and no Superstaq job to use to collect data. Please manually "
                "add results data in order to perform analysis"
            )
        job_status = self.job.status()
        if job_status == "Done":
            self.data = self._collect_device_counts()
            return True
        return False

    @property
    def samples(self) -> Sequence[Sample]:
        """Returns:
        The number of samples used."""
        return self.experiment.samples

    @property
    def num_qubits(self) -> int:
        """Returns:
        The number of qubits in the experiment."""
        return self.experiment.num_qubits

    @property
    def num_circuits(self) -> int:
        """Returns:
        The number of circuits in the experiment."""
        return self.experiment.num_circuits

    def analyze(self, plot_results: bool = True, print_results: bool = True) -> None:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to True.
            print_results: Whether to print the final results. Defaults to True.
        """
        if not self.data_ready:
            warnings.warn(
                "Experiment data is not yet ready to analyse. This is likely because "
                "the Superstaq job has not yet been completed. Either wait and try again "
                "later, or interrogate the `.job` attribute."
            )
            return

        self._analyze()

        if plot_results:
            self.plot_results()

        if print_results:
            self.print_results()

    @abstractmethod
    def _analyze(self) -> None:
        """A method that analyses the `data` attribute and stores the final experimental results."""

    @abstractmethod
    def plot_results(self) -> None:
        """Plot the results of the experiment"""

    @abstractmethod
    def print_results(self) -> None:
        """Prints the key results data."""

    def _collect_device_counts(self) -> pd.DataFrame:
        """Process the counts returned by the server and process into a results dataframe.

        Returns:
            The results dataframe.
        """
        if self.job is None:
            raise ValueError(
                "No Superstaq job associated with these results. Cannot collect device counts."
            )
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

    @property
    def _not_analyzed(self) -> RuntimeError:
        return RuntimeError("Value has not yet been estimated. Please run `.analyze()` method.")


ResultsT = TypeVar("ResultsT", bound=QCVVResults, covariant=True)
# Generic results type for base experiments.


class QCVVExperiment(ABC, Generic[ResultsT]):
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

            results = experiment.run_with_simulator(simulator=sim, <<args/kwargs>>)

    #. Then we analyse the results. If the target was a local simulator this will be available as
       soon as the :code:`run_with_simulator()` method has finished executing. On the other hand
       if a real device was accessed via Superstaq then it may take time for the data to be
       available from the server. The :code:`results.data_ready` attribute will return
       :code:`True` when all data has been collected and is ready to be analyzed.

       .. code::

            if results.data_ready():
                results.analyze(<<args>>)

    When implementing a new experiment, 4 methods need to be implemented:

    #. :meth:`experiment._build_circuits()`: Given a number of circuits and an iterable of the
        different numbers of layers to use, return a list of :class:`Sample` objects that need to
        be sampled during the experiment.

    #. :meth:`results._analyse_results()`: Analyse the experimental data and store the final
        results, for example some fidelities.

    #. :meth:`results.plot_results()`:  Produce any relevant plots that are useful for understanding
        the results of the experiment.

    #. :meth:`results.print_results()`: Prints the results to the console.
    """

    def __init__(
        self,
        num_qubits: int,
        num_circuits: int,
        cycle_depths: Iterable[int],
        *,
        random_seed: int | np.random.Generator | None = None,
        results_cls: type[ResultsT],
        **kwargs: Any,
    ) -> None:
        """Initializes a benchmarking experiment.

        Args:
            num_qubits: The number of qubits used during the experiment. Most subclasses
                will determine this from their other inputs.
            num_circuits: The number of circuits to sample.
            cycle_depths: A sequence of depths to sample.
            random_seed: An optional seed to use for randomization.
            results_cls: The results class to use for the experiment.
            kwargs: Additional kwargs passed to the Superstaq service object.
        """
        self.qubits = cirq.LineQubit.range(num_qubits)
        """The qubits used in the experiment."""

        self.num_circuits = num_circuits
        """The number of circuits to build for each cycle depth."""

        self.cycle_depths = cycle_depths
        """The different cycle depths to test at."""

        self._service_kwargs = kwargs
        """Arguments to pass to the Superstaq service for submitting jobs."""

        self._rng = np.random.default_rng(random_seed)

        self._results_cls: type[ResultsT] = results_cls

        self.samples = self._prepare_experiment()
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
    ) -> Sequence[Sample]:
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

        Returns:
            A sequence of samples for the experiment.
        """

        if any(depth <= 0 for depth in self.cycle_depths):
            raise ValueError("The `cycle_depths` iterator can only include positive values.")

        samples = self._build_circuits(self.num_circuits, self.cycle_depths)
        self._validate_circuits(samples)
        return samples

    def _validate_circuits(self, samples: Sequence[Sample]) -> None:
        """Checks that all circuits contain a terminal measurement of all qubits.

        Args:
            samples: The sequence of samples to check.
        """
        for sample in samples:
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
    ) -> ResultsT:
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

        Returns:
            The experiment results object.
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
    ) -> ResultsT:
        """Use the local simulator to sample the circuits and store the resulting probabilities.

        Args:
            simulator: A local :class:`~cirq.Sampler` to use. If None then the default
                :class:`cirq.Simulator` simulator is used. Defaults to None.
            repetitions: The number of shots to sample. Defaults to 10,000.

        Returns:
            The experiment results object.
        """
        if simulator is None:
            simulator = cirq.Simulator(seed=self._rng)

        records = []
        for sample in tqdm(self.samples, desc="Simulating circuits"):
            result = simulator.run(sample.circuit, repetitions=repetitions)
            hist = result.histogram(key=cirq.measurement_key_name(sample.circuit))
            probabilities = self._canonicalize_probabilities(
                {key: count / sum(hist.values()) for key, count in hist.items()}, self.num_qubits
            )
            records.append({"circuit_index": sample.circuit_index, **sample.data, **probabilities})

        return self._results_cls(
            target="local_simulator",
            experiment=self,
            data=pd.DataFrame(records),
        )

    def run_with_callable(
        self,
        circuit_eval_func: Callable[[cirq.Circuit], dict[str | int, float]],
        **kwargs: Any,
    ) -> ResultsT:
        """Evaluates the probabilities for each circuit using a user provided callable function.
        This function should take a circuit as input and return a dictionary of probabilities for
        each bitstring (including states with zero probability).

        Args:
            circuit_eval_func: The custom function to use when evaluating circuit probabilities.
            kwargs: Additional arguments to pass to the custom function.

        Raises:
            RuntimeError: If the returned probabilities dictionary keys is missing include
                an incorrect number of bits.
            RuntimeError: If the returned probabilities dictionary values do not sum to 1.0.

        Returns:
            The experiment results object.
        """
        records = []
        for sample in tqdm(self.samples, desc="Running circuits"):
            raw_probability = circuit_eval_func(sample.circuit, **kwargs)
            probability = self._canonicalize_probabilities(raw_probability, self.num_qubits)
            records.append({**sample.data, **probability})

        return self._results_cls(
            target="callable",
            experiment=self,
            data=pd.DataFrame(records),
        )

    @staticmethod
    def _canonicalize_bitstring(key: int | str, num_qubits: int) -> str:
        """Checks that the provided key represents a bit string for the given number of qubits.
        If the key is provided as an integer then this is reformatted as a bitstring.

        Args:
            key: The integer or string which represents a bitstring.
            num_qubits: The number of bits that the bitstring needs to have

        Raises:
            ValueError: If the key is integer and negative
            ValueError: If the key is integer but to large for the given number of qubits.
            ValueError: If the key is a string but the wrong length.
            ValueError: If the key is a string but contains characters that are not 0 or 1.

        Returns:
            The canonicalized representation of the bitstring.
        """
        if isinstance(key, int):
            if key < 0:
                raise ValueError(f"The key must be positive. Instead got {key}.")
            if key >= 2**num_qubits:
                raise ValueError(
                    f"The key is too large to be encoded with {num_qubits} qubits. Got {key} "
                    f"but expected less than {2**num_qubits}."
                )
            return format(key, f"0{num_qubits}b")

        if isinstance(key, str):
            if len(key) != num_qubits:
                raise ValueError(
                    f"The key contains the wrong number of bits. Got {len(key)} entries "
                    f"but expected {num_qubits} bits."
                )
            if any(b not in ["0", "1"] for b in key):
                raise ValueError(f"All entries in the bitstring must be 0 or 1. Got {key}.")
            return key

    @staticmethod
    def _canonicalize_probabilities(
        probabilities: dict[str | int, float], num_qubits: int
    ) -> dict[str, float]:
        """Reformats a dictionary of probabilities so that all keys are bitstrings and that
        there are no missing values. Also sorts the dictionary by bitstring.

        Args:
            probabilities: The unformatted probabilities
            num_qubits: The number of qubits, used to determine the bitstring length.

        Raises:
            RuntimeError: If the probabilities do not sum to 1.

        Returns:
            The formatted dictionary of probabilities.
        """
        if not math.isclose(sum(probabilities.values()), 1.0):
            raise RuntimeError(
                f"Provided probabilities do not sum to 1.0. Got {sum(probabilities.values())}."
            )

        new_probability = {
            QCVVExperiment._canonicalize_bitstring(key, num_qubits): val
            for key, val in probabilities.items()
        }

        # Add zero values for any missing bitstrings
        for k in range(2**num_qubits):
            if (bitstring := format(k, f"0{num_qubits}b")) not in new_probability:
                new_probability[bitstring] = 0.0
        # Sort by bitstrings
        new_probability = dict(sorted(new_probability.items()))

        return new_probability
