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
"""Base experiment class and tools used across all experiments."""
from __future__ import annotations

import functools
import numbers
import pathlib
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import cirq
import cirq_superstaq as css
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import supermarq

if TYPE_CHECKING:
    from typing_extensions import Self


def qcvv_resolver(cirq_type: str) -> type[Any] | None:
    """Resolves string's referencing classes in the QCVV library. Used by `cirq.read_json()`
    to deserialize.

    Args:
        cirq_type: The type being resolved

    Returns:
        The corresponding type object (if found) else None

    Raises:
        ValueError: If the provided type is not resolvable
    """
    prefix = "supermarq.qcvv."
    if cirq_type.startswith(prefix):
        name = cirq_type[len(prefix) :]
        if name in supermarq.qcvv.__all__:
            return getattr(supermarq.qcvv, name, None)
    return None


@dataclass
class Sample:
    """A sample circuit to use along with any data about the circuit
    that is needed for analysis
    """

    circuit_realization: int
    """Indicates which realization of the random circuit this sample is. There will be D samples
    with matching circuit realization value, one for each cycle depth being measured. This index is
    useful for grouping results during analysis.
    """
    circuit: cirq.Circuit
    """The raw (i.e. pre-compiled) sample circuit."""
    data: dict[str, Any]
    """The corresponding data about the circuit that is needed when analyzing results
    (e.g. cycle depth)."""

    uuid: uuid.UUID = field(default_factory=uuid.uuid1)
    """The unique ID of the sample."""

    def __hash__(self) -> int:
        return hash(
            (
                self.circuit_realization,
                self.uuid,
                self.circuit.freeze(),
                tuple(sorted(self.data.items())),
            )
        )

    def _json_dict_(self) -> dict[str, Any]:
        """Converts the sample to a json-able dictionary that can be used to recreate the
        sample object.

        Returns:
            Json-able dictionary of the sample data.
        """
        return {
            "circuit": self.circuit,
            "data": self.data,
            "circuit_realization": self.circuit_realization,
            "sample_uuid": str(self.uuid),
        }

    @classmethod
    def _from_json_dict_(
        cls,
        circuit: cirq.Circuit,
        circuit_realization: int,
        data: dict[str, Any],
        sample_uuid: str,
        **_: Any,
    ) -> Self:
        """Creates a sample from a dictionary of the data.

        Args:
            dictionary: Dict containing the sample data.

        Returns:
            The deserialized Sample object.
        """
        return cls(
            circuit=circuit,
            circuit_realization=circuit_realization,
            data=data,
            uuid=uuid.UUID(sample_uuid),
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return "supermarq.qcvv"


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

    def analyze(
        self,
        plot_results: bool = True,
        print_results: bool = True,
        plot_filename: str | None = None,
    ) -> None:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to True.
            print_results: Whether to print the final results. Defaults to True.
            plot_filename: Optional argument providing a filename to save the plots to. Ignored if
                `plot_results=False` Defaults to None, indicating not to save the plot.
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
            self.plot_results(filename=plot_filename)

        if print_results:
            self.print_results()

    @abstractmethod
    def _analyze(self) -> None:
        """A method that analyses the `data` attribute and stores the final experimental results."""

    @abstractmethod
    def plot_results(self, filename: str | None = None) -> plt.Figure:
        """Plot the results of the experiment

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.
        Returns:
            A single matplotlib figure containing the relevant plots of the results data.
        """

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
        _samples: Sequence[Sample] | None = None,
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
            _samples: Optional list of samples to construct the experiment from
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

        if not _samples:
            self.samples = self._prepare_experiment()
        else:
            self.samples = _samples
        """Create all the samples needed for the experiment."""

    def __getitem__(self, key: str | int | uuid.UUID) -> Sample:
        """Gets a sample from the experiment using a key which is either an int (representing the
        index of the circuit) or a str/uuid.UUID (representing the sample's UUID).

        Args:
            key: The key of the sample to find.

        Raises:
            TypeError: If the key is not an int, str or uuid.UUID
            KeyError: If matching Sample can be found
            RuntimeError: If multiple samples are found with the same key.

        Returns:
            The sample corresponding to the key.
        """
        if isinstance(key, numbers.Integral):
            return self.samples[key]
        elif isinstance(key, str):
            key = uuid.UUID(key)
        elif not isinstance(key, uuid.UUID):
            raise TypeError(f"Key must be int, str or uuid.UUID, not {type(key)}")

        matching_samples = [s for s in self.samples if s.uuid == key]
        if len(matching_samples) == 1:
            return matching_samples[0]
        elif len(matching_samples) == 0:
            raise KeyError(f"No sample found with UUID {key}")
        else:
            raise RuntimeError(
                "Multiple samples found with matching key. Something has gone wrong."
            )

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

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
    def canonicalize_bitstring(key: int | str, num_qubits: int) -> str:
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
            TypeError: If the key value is not a string or integral.

        Returns:
            The canonicalized representation of the bitstring.
        """
        if isinstance(key, numbers.Integral):
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

        raise TypeError("Key must either be `numbers.Integral` or `str`.")

    @staticmethod
    def canonicalize_probabilities(
        results: Mapping[str, float] | Mapping[int, float],
        num_qubits: int,
    ) -> dict[str, float]:
        """Reformats a dictionary of probabilities/counts so that all keys are bitstrings and that
        there are no missing values. Also renormalizes so that the resulting probabilities sum to 1
        and sorts the dictionary by bitstring.

        Args:
            results: The unformatted probabilities or counts
            num_qubits: The number of qubits, used to determine the bitstring length.

        Raises:
            ValueError: If any counts or probabilities are negative.
            ValueError: If there are no non-zero counts.

        Returns:
            The formatted dictionary of probabilities.
        """
        if not results:
            return {}

        if any(c < 0 for c in results.values()):
            raise ValueError("Probabilities/counts must be positive.")
        if sum(results.values()) == 0:
            raise ValueError("No non-zero counts.")
        probabilities = {
            QCVVExperiment.canonicalize_bitstring(key, num_qubits): count / sum(results.values())
            for key, count in results.items()
        }
        # Add zero values for any missing bitstrings
        for k in range(2**num_qubits):
            if (bitstring := format(k, f"0{num_qubits}b")) not in probabilities:
                probabilities[bitstring] = 0.0
        # Sort by bitstrings
        probabilities = dict(sorted(probabilities.items()))

        return probabilities

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

    def _map_records_to_samples(
        self, records: Mapping[uuid.UUID | int, Mapping[str, float] | Mapping[int, float]]
    ) -> dict[Sample, Mapping[str, float] | Mapping[int, float]]:
        """Creates a mapping between experiment samples and the provided results records. Records
        with unrecognized sample keys (which should be either an integer index or a UUID) are
        ignored.

        Args:
            records: A mapping of sample keys (either an integer index or a UUID for the sample) to
                the corresponding bitcount/probability results.

        Returns:
            A mapping between experiment samples and the provided results records
        """
        records_not_mapped = dict(records)

        record_mapping: dict[Sample, Mapping[str, float] | Mapping[int, float]] = {}
        for key, record in records.items():
            try:
                sample = self[key]
            except (KeyError, IndexError):  # Ignore any keys that cant be attached to samples
                continue

            if sample in record_mapping:
                raise KeyError(f"Duplicate records found for sample with uuid: {str(sample.uuid)}.")
            record_mapping[sample] = record
            records_not_mapped.pop(key)

        missing_samples = [s for s in self if s not in record_mapping]
        if missing_samples:
            warnings.warn(
                "The following samples are missing records: "
                f"{', '.join(str(s.uuid) for s in missing_samples)}. These will not be included in "
                "the results.",
                stacklevel=2,
            )
        if records_not_mapped:
            warnings.warn(
                f"Unable to find matching sample for {len(records_not_mapped)} record(s).",
                stacklevel=2,
            )

        return record_mapping

    @abstractmethod
    def _json_dict_(self) -> dict[str, Any]:
        """Converts the experiment to a json-able dictionary that can be used to recreate the
        experiment object. Note that the state of the random number generator is not stored.

        .. note:: Must be re-implemented in any subclasses to ensure all important data is stored.

        Returns:
            Json-able dictionary of the experiment data.
        """
        return {
            "cycle_depths": self.cycle_depths,
            "num_circuits": self.num_circuits,
            "num_qubits": self.num_qubits,
            "samples": self.samples,
            **self._service_kwargs,
        }

    @classmethod
    @abstractmethod
    def _from_json_dict_(cls, *args: Any, **kwargs: Any) -> Self:
        """Creates a experiment from an expanded dictionary of the data.

        Returns:
            The deserialized experiment object.
        """

    @classmethod
    def _json_namespace_(cls) -> str:
        return "supermarq.qcvv"

    def to_file(self, filename: str | pathlib.Path) -> None:
        """Save the experiment to a json file.

        Args:
            filename: Filename to save to.
        """
        with open(filename, "w") as file_stream:
            cirq.to_json(self, file_stream)

    @classmethod
    def from_file(cls, filename: str | pathlib.Path) -> Self:
        """Load the experiment from a json file.

        Args:
            filename: Filename to load from.

        Returns:
            The loaded experiment.
        """
        with open(filename, "r") as file_stream:
            experiment = cirq.read_json(
                file_stream,
                resolvers=[*css.SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS, qcvv_resolver],
            )
        return experiment

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
            probabilities = self.canonicalize_probabilities(
                {key: count / sum(hist.values()) for key, count in hist.items()}, self.num_qubits
            )
            records.append(
                {"circuit_realization": sample.circuit_realization, **sample.data, **probabilities}
            )

        return self._results_cls(
            target="local_simulator",
            experiment=self,
            data=pd.DataFrame(records),
        )

    def run_with_callable(
        self,
        circuit_eval_func: Callable[[cirq.Circuit], Mapping[str, float] | Mapping[int, float]],
        **kwargs: Any,
    ) -> ResultsT:
        """Evaluates the probabilities for each circuit using a user provided callable function.
        This function should take a circuit as input and return a dictionary of probabilities for
        each bitstring (including states with zero probability).

        Args:
            circuit_eval_func: The custom function to use when evaluating circuit probabilities.
            kwargs: Additional arguments to pass to the custom function.

        Returns:
            The experiment results object.
        """
        records = []
        for sample in tqdm(self.samples, desc="Running circuits"):
            raw_probability = circuit_eval_func(sample.circuit, **kwargs)
            probability = self.canonicalize_probabilities(raw_probability, self.num_qubits)
            records.append({**sample.data, **probability})

        return self._results_cls(
            target="callable",
            experiment=self,
            data=pd.DataFrame(records),
        )

    def results_from_records(
        self,
        records: Mapping[uuid.UUID | int, Mapping[str, float] | Mapping[int, float]],
    ) -> ResultsT:
        """Creates a results object from records of the counts/probabilities for each sample
        circuit. This function is aimed at users who would like to use the QCVV framework to
        generate sample circuits and analyse the results but need to run these circuits without
        submitting a job to Superstaq.

        Args:
            records: A dictionary of the counts/probabilities for each sample, keyed by either the
                sample UUID or the index of the sample in the experiment. The counts/probabilities
                for each sample should be provided as a
                dictionary of integers or floats (respectively) keyed by either the bitstring or
                the integer value of that bitstring. Note that the distinction between counts and
                probabilities is inferred from the type (int vs float respectively). Please do not
                use float type for counts.

        Returns:
            The experiment results object.
        """
        sample_mapping = self._map_records_to_samples(records)

        results_data = []
        for sample, results in sample_mapping.items():
            probabilities = self.canonicalize_probabilities(
                results,
                self.num_qubits,
            )

            # Add to results data
            results_data.append({**sample.data, **probabilities})

        return self._results_cls(
            target="records",
            experiment=self,
            data=pd.DataFrame(results_data),
        )
