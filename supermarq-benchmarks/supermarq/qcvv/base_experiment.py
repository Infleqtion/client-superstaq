# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import collections
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
import numpy as np
import pandas as pd
from cirq_superstaq.job import JobV3
from tqdm.auto import tqdm

import supermarq

if TYPE_CHECKING:
    from typing import Self

    import matplotlib.pyplot as plt
    from _typeshed import SupportsItems


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
    if cirq_type == "uuid":
        return uuid.UUID

    prefix = "supermarq.qcvv."
    if cirq_type.startswith(prefix):
        name = cirq_type[len(prefix) :]
        if name in supermarq.qcvv.__all__:
            return getattr(supermarq.qcvv, name, None)
    return None


@dataclass
class Sample:
    """A sample circuit to use along with any data about the circuit
    that is needed for analysis.
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
            "uuid": {"cirq_type": "uuid", "hex": str(self.uuid)},
        }

    @classmethod
    def _json_namespace_(cls) -> str:
        return "supermarq.qcvv"


@dataclass
class QCVVResults(ABC):
    """A dataclass for storing the data and analyze results of the experiment. Requires
    subclassing for each new experiment type.
    """

    target: str
    """The target device that was used."""

    experiment: QCVVExperiment[QCVVResults]
    """Reference to the underlying experiment that generated these results experiment."""

    job: css.Job | None = None
    """The associated Superstaq job (if applicable)."""

    data: pd.DataFrame | None = None
    """The raw data generated."""

    _parent: Self | None = None
    _qubits: tuple[cirq.Qid, ...] | None = None

    @property
    def parent(self) -> Self:
        return self._parent or self

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        if self._qubits is None:
            return self.experiment.qubits

        return self._qubits

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
        return len(self.qubits)

    @property
    def num_circuits(self) -> int:
        """Returns:
        The number of circuits in the experiment."""
        return self.experiment.num_circuits

    def __getitem__(self, qubits: cirq.Qid | Sequence[cirq.Qid]) -> Self:
        if not self.data_ready:
            raise ValueError("No results to split.")

        if isinstance(qubits, cirq.Qid):
            qubits = [qubits]
        qubit_indices = [self.experiment.qubits.index(q) for q in qubits]

        num_qubits = self.num_qubits
        bitstrings = [f"{i:0>{num_qubits}b}" for i in range(2**num_qubits)]
        substrings = [f"{i:0>{len(qubits)}b}" for i in range(2 ** len(qubits))]

        substring_map: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        for bitstring in bitstrings:
            substring = "".join(bitstring[qi] for qi in qubit_indices)
            substring_map[substring].append(bitstring)

        assert self.data is not None

        sub_probs = {
            substring: self.data[substring_map[substring]].sum(axis=1) for substring in substrings
        }
        sub_data = pd.DataFrame({**self.data.drop(bitstrings, axis=1), **sub_probs})
        return self.__class__(
            target=self.target,
            experiment=self.experiment,
            data=sub_data,
            _parent=self,
            _qubits=tuple(qubits),
        )

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
                "later, or interrogate the `.job` attribute.",
                stacklevel=2,
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
        """Plot the results of the experiment.

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
        device_counts = self.job.counts()
        records = {sample.uuid: counts for sample, counts in zip(self.samples, device_counts)}
        return self.experiment._structure_records(records)

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
        qubits: int | Sequence[cirq.Qid],
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
            qubits: The qubits used during the experiment. If an integer, this number of line qubits
                will be used. Most subclasses will determine this from their other inputs.
            num_circuits: The number of circuits to sample.
            cycle_depths: A sequence of depths to sample.
            random_seed: An optional seed to use for randomization.
            results_cls: The results class to use for the experiment.
            _samples: Optional list of samples to construct the experiment from
            kwargs: Additional kwargs passed to the Superstaq service object.
        """
        self.qubits: tuple[cirq.Qid, ...]
        if isinstance(qubits, Sequence):
            self.qubits = tuple(qubits)
        else:
            self.qubits = tuple(cirq.LineQubit.range(qubits))

        """The qubits used in the experiment."""

        self.num_circuits = num_circuits
        """The number of circuits to build for each cycle depth."""

        self.cycle_depths = list(cycle_depths)
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
        """The number of qubits used in the experiment."""
        return len(self.qubits)

    @property
    def circuits(self) -> list[cirq.Circuit]:
        """All circuits in this experiment, as a list."""
        return [sample.circuit for sample in self.samples]

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
    def _count_non_barrier_gates(circuit: cirq.Circuit, num_qubits: int | None = None) -> int:
        """Counts the number of gates in a circuit ignoring Barriers. Optionally provide a number
        of qubits in order to only count the number of gates with that number of qubits.

        Args:
            circuit: The circuit to count the gates in.
            num_qubits: Optionally filter gates by the number of qubits they act on.

        Returns:
            The gate count.
        """
        if num_qubits is None:
            return sum(not isinstance(op.gate, css.Barrier) for op in circuit.all_operations())

        return sum(
            (len(op.qubits) == num_qubits and not isinstance(op.gate, css.Barrier))
            for op in circuit.all_operations()
        )

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

    def _interleave_layer(
        self, circuit: cirq.Circuit, layer: cirq.OP_TREE | None, include_final: bool = False
    ) -> cirq.Circuit:
        """Interleave a given operation(s) into a circuit.

        Args:
            circuit: The original circuit.
            layer: The operation(s) to interleave.
            include_final: If True then the interleaving gate is also appended to
                the end of the circuit.

        Returns:
            A copy of the original circuit with the provided layer interleaved.
        """
        if layer:
            layer_circuit = cirq.Circuit(
                css.barrier(*self.qubits),
                cirq.toggle_tags(cirq.Circuit(layer), ("no_compile",)),
                css.barrier(*self.qubits),
            )
        else:
            # If the layer is empty, use a single barrier as a placeholder
            layer_circuit = cirq.Circuit(css.barrier(*self.qubits))

        interleaved_circuit = circuit.copy()
        interleaved_circuit.batch_insert(
            [(k, layer_circuit) for k in range(len(circuit) - int(not include_final), 0, -1)]
        )
        return interleaved_circuit

    def _map_records_to_samples(
        self, records: SupportsItems[uuid.UUID | int, Mapping[str, float] | Mapping[int, float]]
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
        record_mapping: dict[Sample, Mapping[str, float] | Mapping[int, float]] = {}
        num_unmatched = 0
        for key, record in records.items():
            try:
                sample = self[key]
            except (KeyError, IndexError):  # Ignore any keys that cant be attached to samples
                num_unmatched += 1
                continue

            if sample in record_mapping:
                raise KeyError(f"Duplicate records found for sample with uuid: {sample.uuid!s}.")
            record_mapping[sample] = record

        missing_samples = [s for s in self if s not in record_mapping]
        if missing_samples:
            warnings.warn(
                "The following samples are missing records: "
                f"{', '.join(str(s.uuid) for s in missing_samples)}. These will not be included in "
                "the results.",
                stacklevel=2,
            )
        if num_unmatched:
            warnings.warn(
                f"Unable to find matching sample for {num_unmatched} record(s).",
                stacklevel=2,
            )

        return record_mapping

    def _structure_records(
        self, records: SupportsItems[uuid.UUID | int, Mapping[str, float] | Mapping[int, float]]
    ) -> pd.DataFrame:
        """Constructs a `pandas.DataFrame` from the provided records.

        Args:
            records: A dictionary of the counts/probabilities for each sample, keyed by either the
                sample UUID or the index of the sample in the experiment. The counts/probabilities
                for each sample should be provided as a dictionary of keyed by either the bitstring
                or the integer value of that bitstring.

        Returns:
            A `DataFrame` containing the provided counts and corresponding sample information.
        """
        sample_mapping = self._map_records_to_samples(records)

        results_data = []
        for sample, results in sample_mapping.items():
            probabilities = self.canonicalize_probabilities(results, self.num_qubits)

            # Add to results data
            result = {
                "uuid": sample.uuid,
                "circuit_realization": sample.circuit_realization,
                **sample.data,
                **probabilities,
            }
            results_data.append(result)

        return pd.DataFrame(results_data)

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
            "qubits": self.qubits,
            "_samples": self.samples,
            **self._service_kwargs,
        }

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
        with open(filename) as file_stream:
            experiment = cirq.read_json(
                file_stream,
                resolvers=[*css.SUPERSTAQ_RESOLVERS, *cirq.DEFAULT_RESOLVERS, qcvv_resolver],
            )
        return experiment

    def _prepare_experiment(
        self,
    ) -> Sequence[Sample]:
        """Prepares the circuits needed for the experiment.

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

        if isinstance(experiment_job, JobV3):  # pragma: no cover
            raise NotImplementedError(
                "QCVV experiments are not using v0.3.0 of the Superstaq API yet."
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

        records: dict[uuid.UUID, dict[int, int]] = {}
        for sample in tqdm(self.samples, desc="Simulating circuits"):
            result = simulator.run(sample.circuit, repetitions=repetitions)
            records[sample.uuid] = result.histogram(key=cirq.measurement_key_name(sample.circuit))

        data = self._structure_records(records)
        return self._results_cls(target="local_simulator", experiment=self, data=data)

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
        records: dict[uuid.UUID, Mapping[str, float] | Mapping[int, float]] = {}
        for sample in tqdm(self.samples, desc="Running circuits"):
            raw_probability = circuit_eval_func(sample.circuit, **kwargs)
            records[sample.uuid] = raw_probability

        data = self._structure_records(records)
        return self._results_cls(target="callable", experiment=self, data=data)

    def results_from_records(
        self, records: SupportsItems[uuid.UUID | int, Mapping[str, float] | Mapping[int, float]]
    ) -> ResultsT:
        """Creates a results object from records of the counts/probabilities for each sample
        circuit. This function is aimed at users who would like to use the QCVV framework to
        generate sample circuits and analyse the results but need to run these circuits without
        submitting a job to Superstaq.

        Args:
            records: A dictionary of the counts/probabilities for each sample, keyed by either the
                sample UUID or the index of the sample in the experiment. The counts/probabilities
                for each sample should be provided as a dictionary of keyed by either the bitstring
                or the integer value of that bitstring.

        Returns:
            The experiment results object.
        """
        data = self._structure_records(records)
        return self._results_cls(target="records", experiment=self, data=data)
