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
"""Tooling for SU(2) benchmarking
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cirq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
from tqdm.contrib.itertools import product

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, BenchmarkingResults, Sample


@dataclass(frozen=True)
class SU2Results(BenchmarkingResults):
    """Data structure for the SU2 experiment results."""

    two_qubit_gate_fidelity: float
    """Estimated two qubit gate fidelity"""
    two_qubit_gate_fidelity_std: float
    """Standard deviation of the two qubit gate fidelity estimate"""
    single_qubit_noise: float
    single_qubit_noise_std: float

    experiment_name = "SU2"

    @property
    def two_qubit_gate_error(self) -> float:
        """Returns:
        The two qubit gate error. Equal to one minus the fidelity.
        """
        return 1 - self.two_qubit_gate_fidelity

    @property
    def two_qubit_gate_error_std(self) -> float:
        """Returns:
        The two qubit gate error standard deviation. Equal to standard deviation of the
        fidelity.
        """
        return self.two_qubit_gate_fidelity_std


class SU2(BenchmarkingExperiment[SU2Results]):
    r"""SU2 benchmarking experiment.

    SU2 benchmarking extracts the fidelity of a given two qubit gate, even in the presence of
    additional single qubit errors. The method works by sampling circuits of the form

    .. code::

        0: ──|─Rr───Q───X───Q──|─ ^{n} ... ─|─Rr───X─|─ ^{N-n} ... ──Rf───M───
             |      │       │  |            |        |                    │
        1: ──|─Rr───Q───X───Q──|─      ... ─|─Rr───X─|─        ... ──Rf───M───

    Where each :code:`Rr` gate is a randomly chosen :math:`SU(2)` rotation and the :code:`Rf` gates
    are single qubit :math:`SU(2)` rotations that in the absence of noise invert the preceding
    circuit so that the final qubit state should be :code:`00`.

    An exponential fit decay is then fitted to the observed 00 state probability as it decays with
    the number of two qubit gates included. Note that all circuits contain a fixed number of single
    qubit gates, so that the contribution for single qubit noise is constant.

    See Fig. 3 of :ref:`https://www.nature.com/articles/s41586-023-06481-y#Fig3` for further
    details.
    """

    def __init__(
        self,
        two_qubit_gate: cirq.Gate = cirq.CZ,
    ) -> None:
        """Args:
        two_qubit_gate: The Clifford gate to measure the gate error of.
        num_qubits: The number of qubits to experiment on. Must equal 2.
        """
        super().__init__(num_qubits=2)

        if two_qubit_gate.num_qubits() != 2:
            raise ValueError(
                "The `two_qubit_gate` parameter must be a gate that acts on exactly two qubits."
            )
        self.two_qubit_gate = two_qubit_gate
        """The two qubit gate to be benchmarked"""

    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
    ) -> Sequence[Sample]:
        """Build a list of circuits required for the experiment.

        These circuits are stored in :class:`Sample` objects along with any additional data that is
        needed during the analysis.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different cycle depths to use during the experiment.

        Returns:
           The list of experiment samples.
        """
        samples = []
        max_depth = max(cycle_depths)
        for depth, _ in product(cycle_depths, range(num_circuits), desc="Building circuits"):
            circuit = cirq.Circuit(
                *[self._component(include_two_qubit_gate=True) for _ in range(depth)],
                *[self._component(include_two_qubit_gate=False) for _ in range(max_depth - depth)],
            )
            circuit_inv = cirq.inverse(circuit)
            # Decompose circuit inverse into a pair of single qubit rotation gates
            _, rot_1, rot_2 = cirq.kron_factor_4x4_to_2x2s(cirq.unitary(circuit_inv))

            if (op_1 := cirq.single_qubit_matrix_to_phxz(rot_1)) is not None:
                circuit += op_1(self.qubits[0])

            if (op_2 := cirq.single_qubit_matrix_to_phxz(rot_2)) is not None:
                circuit += op_2(self.qubits[1])

            circuit += cirq.measure(sorted(circuit.all_qubits()))

            samples.append(Sample(raw_circuit=circuit, data={"num_two_qubit_gates": 2 * depth}))
        return samples

    def _process_probabilities(self, samples: Sequence[Sample]) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into a data frame
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """
        records = []
        for sample in samples:
            records.append(
                {
                    "num_two_qubit_gates": sample.data["num_two_qubit_gates"],
                    **sample.probabilities,
                }
            )

        return pd.DataFrame(records)

    def analyze_results(self, plot_results: bool = True) -> SU2Results:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """
        fit = linregress(
            x=self.raw_data["num_two_qubit_gates"],
            y=np.log(self.raw_data["00"] - 1 / 4),
            # Scale the y coordinate to account for limit of the decay being 1/4
        )
        gate_fid = np.exp(fit.slope)
        gate_fid_std = fit.stderr * gate_fid

        single_qubit_noise = 1 - 4 / 3 * np.exp(fit.intercept)
        single_qubit_noise_std = fit.intercept_stderr * (1 - single_qubit_noise)

        self._results = SU2Results(
            target="& ".join(self.targets),
            total_circuits=len(self.samples),
            two_qubit_gate_fidelity=gate_fid,
            two_qubit_gate_fidelity_std=gate_fid_std,
            single_qubit_noise=single_qubit_noise,
            single_qubit_noise_std=single_qubit_noise_std,
        )

        if plot_results:
            self.plot_results()

        return self.results

    @staticmethod
    def _haar_random_rotation() -> cirq.Gate:
        """Returns:
        Haar randomly sampled SU(2) rotation.
        """
        gate: cirq.Gate | None = None
        while gate is None:
            gate = cirq.single_qubit_matrix_to_phxz(cirq.testing.random_special_unitary(dim=2))
        return gate

    def _component(self, include_two_qubit_gate: bool) -> cirq.Circuit:
        """Core component of the experimental circuits.

        These circuits that are repeated to create the full circuit. Can optionally include the
        two qubit gate being measured, as is required for the
        first half of the full circuit, but not for the second half.

        The component looks like:
        .. code::

            0: ───R1───Q───X───Q───
                       │       │
            1: ───R2───Q───X───Q───

        where :code:`R1` and :code:`R2` are Haar randomly chosen SU(2) rotation
        and :code:`Q-Q` represents the two qubit gate being measured.

        Args:
            include_two_qubit_gate: Whether to include the two qubit gate being measured

        Returns:
            The sub circuit to be repeated when building the full circuit
        """
        return cirq.Circuit(
            self._haar_random_rotation().on(self.qubits[0]),
            self._haar_random_rotation().on(self.qubits[1]),
            (
                self.two_qubit_gate(*self.qubits).with_tags("no_compile")
                if include_two_qubit_gate
                else []
            ),
            cirq.X.on_each(*self.qubits),
            (
                self.two_qubit_gate(*self.qubits).with_tags("no_compile")
                if include_two_qubit_gate
                else []
            ),
        )

    def plot_results(self) -> None:
        """Plot the results of the experiment"""
        _, ax = plt.subplots()
        sns.scatterplot(
            data=self.raw_data.melt(
                id_vars="num_two_qubit_gates", var_name="state", value_name="prob"
            ),
            x="num_two_qubit_gates",
            y="prob",
            hue="state",
            hue_order=["00", "01", "10", "11"],
            style="state",
            ax=ax,
        )
        ax.plot(
            xx := self.raw_data["num_two_qubit_gates"],
            3 / 4 * (1 - self.results.single_qubit_noise) * self.results.two_qubit_gate_fidelity**xx
            + 0.25,
            label="00 (fit)",
        )
        ax.set_xlabel("Number of two qubit gates")
        ax.set_ylabel("State probability")
        ax.legend(title="State")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))