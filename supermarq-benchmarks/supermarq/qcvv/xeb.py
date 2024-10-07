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
"""Tooling for cross entropy benchmark experiments.
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cirq
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm.auto
import tqdm.contrib.itertools

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, BenchmarkingResults, Sample


@dataclass
class XEBSample(Sample):
    """The samples used in XEB experiments."""

    target_probabilities: dict[str, float] | None = None
    """The target probabilities obtained through a noiseless simulator"""
    sample_probabilities: dict[str, float] | None = None
    """The sample probabilities obtained from the chosen target"""

    def sum_target_probs_square(self) -> float:
        """Compute the sum of the squared target probabilities.

        Raises:
            RuntimeError: If no target probabilities have been initialised.

        Returns:
            float: The sum of squared target probabilities.
        """
        if self.target_probabilities is None:
            raise RuntimeError("`target_probabilities` have not yet been initialised")

        return sum(prob**2 for prob in self.target_probabilities.values())

    def sum_target_cross_sample_probs(self) -> float:
        """Compute the dot product between the sample and target probabilities

        Raises:
            RuntimeError: If either the target or sample probabilities have not yet been
                initialised.

        Returns:
            float: The dot product between the sample and target probabilities.
        """
        if self.target_probabilities is None:
            raise RuntimeError("`target_probabilities` have not yet been initialised")

        if self.sample_probabilities is None:
            raise RuntimeError("`sample_probabilities` have not yet been initialised")

        return sum(
            self.target_probabilities[state] * self.sample_probabilities[state]
            for state in self.target_probabilities
        )


@dataclass(frozen=True)
class XEBResults(BenchmarkingResults):
    """Results from an XEB experiment."""

    cycle_fidelity_estimate: float
    """Estimated cycle fidelity."""
    cycle_fidelity_estimate_std: float
    """Standard deviation for the cycle fidelity estimate."""

    experiment_name = "XEB"


class XEB(BenchmarkingExperiment[XEBResults]):
    r"""Cross-entropy benchmarking (XEB) experiment.

    The XEB experiment can be used to estimate the combined fidelity of a repeating
    cycle of gates. In our case, where we restrict ourselves to two qubits, we use
    cycles made up of two randomly selected single qubit phased XZ gates and a constant
    two qubit gate. This is illustrated as follows:

    For each randomly generated circuit, with a given number of cycle, we compare the
    simulated state probabilities, :math:`p(x)` with those achieved by running the circuit
    on a given target, :math:`\hat{p}(x)`. The fidelity of a circuit containing :math:`d`
    cycles, :math:`f_d` can then be estimated as

    .. math::

        \sum_{x \in \{0, 1\}^n} p(x) \hat{p}(x) - \frac{1}{2^n} =
        f_d \left(\sum_{x \in \{0, 1\}^n} p(x)^2 -  \frac{1}{2^n}\right)

    We can therefore fit a linear model to estimate the value of :math:`f_d`. We the estimate
    the fidelity of the cycle, :math:`f_{\mathrm{cycle}}` as

    .. math::

        f_d = A(f_{cycle})^d

    Thus fitting another linear model to :math:`\log(f_d) \sim d` provides us with an estimate
    of the cycle fidelity.

    For more details see: https://www.nature.com/articles/s41586-019-1666-5
    """

    def __init__(
        self,
        single_qubit_gate_set: list[cirq.Gate] | None = None,
        two_qubit_gate: cirq.Gate | None = cirq.CZ,
        *,
        random_seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initializes a cross-entropy benchmarking experiment.

        Args:
            single_qubit_gate_set: Optional list of single qubit gates to randomly sample from when
                generating random circuits. If not provided defaults to phased XZ gates with 1/4 pi
                intervals.
            two_qubit_gate: The two qubit gate to interleave between the single qubit gates. If None
                then no two qubit gate is used. Defaults to control-Z gate.
            random_seed: An optional seed to use for randomization.
        """
        super().__init__(num_qubits=2, random_seed=random_seed)

        self._circuit_fidelities: pd.DataFrame | None = None

        self._samples: Sequence[XEBSample] | None = None  # Overwrite with modified sampled object

        self.two_qubit_gate: cirq.Gate | None = two_qubit_gate
        """The two qubit gate to use for interleaving."""

        self.single_qubit_gate_set: list[cirq.Gate]
        """The single qubit gates to randomly sample from"""

        if single_qubit_gate_set is None:
            gate_exponents = np.linspace(start=0, stop=7 / 4, num=8)
            self.single_qubit_gate_set = [
                cirq.PhasedXZGate(
                    z_exponent=z,  # 1) Choose an axis in the xy-plane, zπ from the +x-axis.
                    x_exponent=0.5,  # 2) Rotate about the axis in 1) by a fixed π/2.
                    axis_phase_exponent=a,  # 3) Rotate about the +z-axis by aπ (a final phasing).
                )
                for a, z in itertools.product(
                    gate_exponents, repeat=2
                )  # enumerates every possible (a, z)
            ]
        else:
            self.single_qubit_gate_set = single_qubit_gate_set

    ##############
    # Properties #
    ##############
    @property
    def circuit_fidelities(self) -> pd.DataFrame:
        """The circuit fidelity calculations from the most recently run experiment.

        Raises:
            RuntimeError: If no data is available.
        """
        if self._circuit_fidelities is None:
            raise RuntimeError("No data to retrieve. The experiment has not been run.")

        return self._circuit_fidelities

    @property
    def samples(self) -> Sequence[XEBSample]:  # Overwrite with XEBSample return type
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
    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
    ) -> Sequence[XEBSample]:
        """Build a list of random circuits to perform the XEB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.

        Returns:
            The list of experiment samples.
        """
        random_circuits = []
        for _, depth in tqdm.contrib.itertools.product(
            range(num_circuits), cycle_depths, desc="Building circuits"
        ):
            num_single_qubit_gate_layers = depth + int(self.two_qubit_gate is not None)
            chosen_single_qubit_gates = self._rng.choice(
                np.asarray(self.single_qubit_gate_set),
                size=(num_single_qubit_gate_layers, self.num_qubits),
            )

            circuit = cirq.Circuit(
                gate.on(qubit)
                for gates_in_layer in chosen_single_qubit_gates
                for gate, qubit in zip(gates_in_layer, self.qubits)
            )

            if self.two_qubit_gate is not None:
                circuit = self._interleave_op(circuit, self.two_qubit_gate(*self.qubits))

            random_circuits.append(
                XEBSample(
                    raw_circuit=circuit + cirq.measure(sorted(circuit.all_qubits())),
                    data={
                        "circuit_depth": len(circuit),
                        "num_cycles": depth,
                        "two_qubit_gate": str(self.two_qubit_gate),
                    },
                )
            )

        return random_circuits

    def _process_probabilities(
        self, samples: Sequence[XEBSample]  # type: ignore[override]
    ) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into the data structures
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """
        samples = list(samples)
        missing_count = 0
        for sample in samples:
            if sample.probabilities is None:
                missing_count += 1
                samples.remove(sample)
            else:
                sample.sample_probabilities = sample.probabilities
                sample.probabilities = {}
        if missing_count > 0:
            warnings.warn(
                f"{missing_count} sample(s) are missing `probabilities`. "
                "These samples have been omitted."
            )

        for sample in tqdm.auto.tqdm(samples, desc="Evaluating circuits"):
            sample.target_probabilities = self._simulate_sample(sample)

        records = []
        for sample in samples:
            if sample.sample_probabilities is not None and sample.target_probabilities is not None:
                target_probabilities = {
                    f"p({key})": value for key, value in sample.target_probabilities.items()
                }
                sample_probabilities = {
                    f"p^({key})": value for key, value in sample.sample_probabilities.items()
                }
                records.append(
                    {
                        "cycle_depth": sample.data["num_cycles"],
                        "circuit_depth": sample.data["circuit_depth"],
                        **target_probabilities,
                        **sample_probabilities,
                        "sum_p(x)p(x)": sample.sum_target_probs_square(),
                        "sum_p(x)p^(x)": sample.sum_target_cross_sample_probs(),
                    }
                )
        return pd.DataFrame(records)

    def _simulate_sample(self, sample: XEBSample) -> dict[str, float]:
        """Simulates the exact probabilities of measuring all possible bitstrings
        with a given sample.

        Args:
            sample: The sample to simulate.

        Returns:
            A dictionary of the probability of each bitstring.
        """
        sim = cirq.Simulator(seed=self._rng)

        result = sim.simulate(
            cirq.drop_terminal_measurements(sample.circuit),
            qubit_order=sorted(sample.circuit.all_qubits()),
        )
        return {
            f"{i:0{self.num_qubits}b}": np.abs(amp) ** 2
            for i, amp in enumerate(result.final_state_vector)
        }

    ###################
    # Public Methods  #
    ###################
    def analyze_results(self, plot_results: bool = True) -> XEBResults:
        """Analyse the results and calculate the estimated circuit fidelity.

        Args:
            plot_results (optional): Whether to generate the data plots. Defaults to True.

        Returns:
           The final results from the experiment.
        """

        # Fit a linear model for each cycle depth to estimate the circuit fidelity
        records = []
        for depth in set(self.raw_data.cycle_depth):
            df = self.raw_data[self.raw_data.cycle_depth == depth]
            fit = scipy.stats.linregress(
                x=df["sum_p(x)p(x)"] - 1 / 2**self.num_qubits,
                y=df["sum_p(x)p^(x)"] - 1 / 2**self.num_qubits,
            )

            records.append(
                {
                    "cycle_depth": depth,
                    "circuit_fidelity_estimate": fit.slope,
                    "circuit_fidelity_estimate_std": fit.stderr,
                }
            )

        self._circuit_fidelities = pd.DataFrame(records)

        self.circuit_fidelities["log_fidelity_estimate"] = np.log(
            self.circuit_fidelities.circuit_fidelity_estimate
        )

        # Fit a linear model to the depth ~ log(fidelity) to approximate the cycle fidelity
        cycle_fit = scipy.stats.linregress(
            x=self.circuit_fidelities.cycle_depth,
            y=np.log(self.circuit_fidelities.circuit_fidelity_estimate),
        )
        cycle_fidelity_estimate = np.exp(cycle_fit.slope)
        cycle_fidelity_estimate_std = cycle_fidelity_estimate * cycle_fit.stderr

        self._results = XEBResults(
            target="$ ".join(self.targets),
            total_circuits=len(self.samples),
            cycle_fidelity_estimate=cycle_fidelity_estimate,
            cycle_fidelity_estimate_std=cycle_fidelity_estimate_std,
        )

        if plot_results:
            self.plot_results()

        return self.results

    def plot_results(self) -> None:
        """Plot the experiment data and the corresponding fits."""
        plot_1 = sns.lmplot(
            data=self.raw_data,
            x="sum_p(x)p(x)",
            y="sum_p(x)p^(x)",
            hue="cycle_depth",
            palette="dark:r",
            legend="full",
            ci=None,
        )
        sns.move_legend(plot_1, "center right")
        ax_1 = plot_1.axes.item()
        plot_1.tight_layout()
        ax_1.set_xlabel(r"$\sum p(x)^2$", fontsize=15)
        ax_1.set_ylabel(r"$\sum p(x) \hat{p}(x)$", fontsize=15)
        ax_1.set_title(r"Linear fit per cycle depth", fontsize=15)

        plot_2 = sns.lmplot(
            data=self.circuit_fidelities,
            x="cycle_depth",
            y="circuit_fidelity_estimate",
            hue="cycle_depth",
            palette="dark:r",
        )
        ax_2 = plot_2.axes.item()
        plot_2.tight_layout()
        ax_2.set_xlabel(r"Cycle depth", fontsize=15)
        ax_2.set_ylabel(r"Circuit fidelity", fontsize=15)
        ax_2.set_title(r"Exponential decay of circuit fidelity", fontsize=15)

        # Add fit line
        x = np.linspace(
            self.circuit_fidelities.cycle_depth.min(), self.circuit_fidelities.cycle_depth.max()
        )
        y = self.results.cycle_fidelity_estimate**x
        y_p = (self.results.cycle_fidelity_estimate + self.results.cycle_fidelity_estimate_std) ** x
        y_m = (self.results.cycle_fidelity_estimate - self.results.cycle_fidelity_estimate_std) ** x
        ax_2.plot(x, y, color="tab:red", linewidth=2)
        ax_2.fill_between(x, y_m, y_p, alpha=0.2, color="tab:red")
