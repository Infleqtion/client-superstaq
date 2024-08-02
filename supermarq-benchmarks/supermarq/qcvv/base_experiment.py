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

import pprint
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


@dataclass
class Sample:
    """A sample circuit to use along with any data about the circuit
    that is needed for analysis
    """

    circuit: cirq.Circuit
    """The sample circuit."""
    data: dict[str, Any]
    """The corresponding data about the circuit"""
    probabilities: dict[str, float] = field(init=False)
    """The probabilities of the computational basis states"""
    job: css.Job | None = None
    """The superstaq job corresponding to the sample. Defaults to None if no job is
    associated with the sample."""

    @property
    def target(self) -> str:
        """Returns:
        The name of the target that the sample was submitted to
        """
        if self.job is not None:
            # If there is a job then get the target
            return self.job.target()

        if hasattr(self, "probabilities"):
            # If no job, but probabilities have been calculated, infer that a local
            # simulator was used.
            return "Local simulator"

        # Otherwise the experiment hasn't yet been run so there is no target.
        return "No target"


@dataclass(frozen=True)
class QCVVResults:
    """A dataclass for storing the results of the experiment. Requires subclassing for
    each new experiment type"""

    experiment_name: str
    """The name of the experiment."""
    target: str
    """The target device that was used."""
    total_circuits: int
    """The total number of circuits used in the experiment."""


class BenchmarkingExperiment(ABC):
    """Base class for gate benchmarking experiments.

    The interface for implementing these experiments is as follows:

    #. First instantiate the desired experiment object

        .. code::

            experiment = ExampleExperiment(<<args/kwargs>>)

    #. Run the experiment on the desired target. This can either be a custom simulator
       or a real device name. For example

        .. code::

            noise_model = cirq.depolarize(p=0.01, n_qubits=1)
            target = cirq.DensityMatrixSimulator(noise=noise_model)

            experiment.run(target=target, <<args/kwargs>>)

    #. Then we analyse the results. If the target was a local simulator this will be available as
       soon as the :code:`run()` method has finished executing. On the other hand if a real device
       was accessed via Superstaq then it may take time for the data to be available from the
       server. The :code:`collect_data()` will return :code:`True` when all data has been collected
       and is ready to be analysed.

       .. code::
            if self.collect_data():
                results = experiment.analyse_results(<<args>>)

    #. The final results of the experiment will be stored in the :code:`results` attribute as a
       :class:`QCVVResults` of values, while all the data from the experiment will be
       stored in the :code:`raw_data` attribute as a :class:`~pandas.DataFrame`. Some experiments
       may include additional data attributes for data generated during the analysis.

        .. code::

            results = experiment.results
            data = experiment.raw_data

    .. warning::
        Note that each time the :code:`run()` method is called the
        previous jobs, results and data are overwritten.

    When implementing a new experiment, 4 methods need to be implemented:

    #. :meth:`build_circuits`: Given a number of circuits and an iterable of the different numbers
        of layers to use, return a list of :class:`Sample` objects that need to be sampled during
        the experiment.

    #. :meth:`process_probabilities`: Take the probability distribution over the
        computational basis resulting from running each circuit and combine the relevant details
        into a :class:`pandas.DataFrame`.

    #. :meth:`analyse_results`: Analyse the data in the :attr:`raw_data` dataframe and return a
        :class:`QCVVResults` object containing the results of the experiment.

    #. :meth:`plot_results`: Produce any relevant plots that are useful for understanding the
        results of the experiment.

    Additionally the :class:`QCVVResults` dataclass needs subclassing to hold the specific results
    of the new experiment.

    """

    def __init__(
        self,
        num_qubits: int,
        **kwargs: Any,
    ) -> None:
        """Args:
        num_qubits: The number of qubits used during the experiment. Most subclasses
            will determine this from their other inputs.
        kwargs: Additional kwargs passed to the Superstaq service object.
        """
        self.qubits = cirq.LineQubit.range(num_qubits)
        """The qubits used in the experiment."""

        self._raw_data: pd.DataFrame | None = None
        "The data generated during the experiment"

        self._results: QCVVResults | None = None
        """The attribute to store the results in."""

        self._samples: Sequence[Sample] | None = None
        """The attribute to store the experimental samples in."""

        self._service: css.service.Service = css.service.Service(**kwargs)
        """The superstaq service for submitting jobs."""

    @property
    def results(self) -> QCVVResults:
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
    def raw_data(self) -> pd.DataFrame:
        """The data from the most recently run experiment.

        Raises:
            RuntimeError: If no results are available.
        """
        if self._raw_data is None:
            raise RuntimeError("No data to retrieve. The experiment has not been run.")

        return self._raw_data

    @property
    def num_qubits(self) -> int:
        """Returns:
        The number of qubits used in the experiment
        """
        return len(self.qubits)

    @property
    def sample_targets(self) -> list[str]:
        """Returns:
        A list of the unique target that each sample was submitted to
        """
        return sorted(set(sample.target for sample in self.samples))

    def _validate_circuits(self) -> None:
        """Checks that all circuits contain a terminal measurement of all qubits."""
        for sample in self.samples:
            if not sample.circuit.are_all_measurements_terminal():
                raise ValueError("QCVV experiment circuits can only contain terminal measurements")
            if not sorted(sample.circuit[-1].qubits) == sorted(self.qubits):
                raise ValueError(
                    "The terminal measurement in QCVV experiment circuits must measure all qubits."
                )

    def _prepare_experiment(
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

        self._samples = self.build_circuits(num_circuits, cycle_depths)
        self._validate_circuits()

    def run_on_device(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
        target: str,
        shots: int = 10_000,
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
            num_circuits: Number of circuits to run.
            cycle_depths: An iterable of the different layer depths to use during the experiment.
            target: The name of a Superstaq target.
            shots: The number of shots to sample. Defaults to 10,000.
            method: Optional method to use on the Superstaq device. Defaults to None corresponding
                to normal running.
            target_options: Optional configuration dictionary passed when submitting the job.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.

        Return:
            The superstaq job containing all the circuits submitted as part of the experiment.
        """
        self._prepare_experiment(num_circuits, cycle_depths, overwrite)

        experiment_job = self._service.create_job(
            [sample.circuit for sample in self.samples],
            target=target,
            method=method,
            repetitions=shots,
            **target_options,
        )
        for k, sample in enumerate(self.samples):
            sample.job = experiment_job[k]

        return experiment_job

    def sample_statuses(self) -> list[str | None]:
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

    def retrieve_ss_jobs(self) -> dict[str, str]:
        """Retrieve the jobs from the server.

        Returns:
            A dictionary of the statuses of any jobs that have not been successfully completed.
        """
        # If no jobs then return empty dictionary
        if not [sample for sample in self.samples if sample.job is not None]:
            return {}

        statuses = {}
        waiting_samples = [
            sample for sample in self.samples if not hasattr(sample, "probabilities")
        ]
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

    def run_with_simulator(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
        target: cirq.SimulatorBase | None = None,  # type: ignore [type-arg]
        shots: int = 10_000,
        overwrite: bool = False,
    ) -> None:
        """Use the local simulator to sample the circuits and store the resulting probabilities.

        Args:
            num_circuits: Number of circuits to run.
            cycle_depths: An iterable of the different layer depths to use during the experiment.
            target: A local :class:`~cirq.SimulatorBase` to use. If None then the default
                :class:`cirq.Simulator` simulator is used. Defaults to None.
            shots: The number of shots to sample. Defaults to 10,000.
            overwrite: Whether to force an experiment run even if there is existing data that would
                be over written in the process. Defaults to False.
        """
        self._prepare_experiment(num_circuits, cycle_depths, overwrite)

        if target is None:
            target = cirq.Simulator()

        for sample in tqdm(self.samples, desc="Simulating circuits"):
            # Use transpose (.T) to reshape samples output from (n x 1) into (1 x n)
            result = target.run(sample.circuit, repetitions=shots)
            hist = result.histogram(key=cirq.measurement_key_name(sample.circuit))
            sample.probabilities = {
                f"{i:0{self.num_qubits}b}": count / shots for i, count in hist.items()
            }

    def collect_data(self, force: bool = False) -> bool:
        """Collect the data from the samples and process it into the :attr:`raw_data` attribute.

        If the data is successfully stored in the :attr:`raw_data` attribute then the function will
        return :code:`True`

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
        outstanding_statuses = self.retrieve_ss_jobs()
        if outstanding_statuses:
            print(
                "Not all circuits have been sampled. "
                "Please wait and try again.\n"
                f"Outstanding Superstaq jobs:\n{pprint.pformat(outstanding_statuses)}"
            )
            if not force:
                return False

        completed_samples = [sample for sample in self.samples if hasattr(sample, "probabilities")]

        if not len(completed_samples) == len(self.samples):
            print("Some samples do not have probability results.")
            if not force:
                return False

        if len(completed_samples) == 0:
            raise RuntimeError("Cannot force data collection when there are no completed samples.")

        self._raw_data = self.process_probabilities(completed_samples)
        return True

    @abstractmethod
    def build_circuits(
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
            The list of circuit objects
        """

    @abstractmethod
    def process_probabilities(self, samples: Sequence[Sample]) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into a data frame
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """

    @abstractmethod
    def plot_results(self) -> None:
        """Plot the results of the experiment"""

    @abstractmethod
    def analyse_results(self, plot_results: bool = True) -> QCVVResults:
        """Perform the experiment analysis and store the results in the `results` attribute

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """

    def _state_probs_to_dict(
        self, probs: np.typing.NDArray[np.float64], prefix: str = "", suffix: str = ""
    ) -> dict[str, float]:
        """Converts a numpy array of state probabilities to a dictionary indexed
        by the bitstring. Optional prefix and suffix can be added.

        Args:
            probs: Numpy array of coefficients.
            prefix: Optional prefix to the bitstring key. Defaults to "".
            suffix: Optional suffix to the bitstring key. Defaults to "".

        Returns:
            Dictionary of state probabilities indexed by bitstring.
        """
        return {
            prefix + format(idx, f"0{self.num_qubits}b") + suffix: coefficient
            for idx, coefficient in enumerate(probs)
        }

    @staticmethod
    def _interleave_gate(
        circuit: cirq.Circuit, gate: cirq.Gate, include_final: bool = False
    ) -> cirq.Circuit:
        """Interleave a given gate into a circuit.

        Args:
            circuit: The original circuit.
            gate: The gate to interleave.
            include_final: If True then the interleaving gate is also appended to
                the end of the circuit.

        Returns:
            A copy of the original circuit with the provided gate interleaved.
        """
        qubits = sorted(circuit.all_qubits())
        interleaved_circuit = circuit.copy()
        interleaved_circuit.batch_insert(
            [(k, gate(*qubits)) for k in range(len(circuit) - int(not include_final), 0, -1)]
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
