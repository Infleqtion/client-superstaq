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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cirq
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm.auto
import tqdm.contrib.itertools

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample


@dataclass
class XEBResults(QCVVResults):
    """Results from an XEB experiment."""

    _circuit_fidelities: pd.DataFrame | None = None
    _cycle_fidelity_estimate: float | None = None
    """Estimated cycle fidelity."""
    _cycle_fidelity_estimate_std: float | None = None
    """Standard deviation for the cycle fidelity estimate."""

    @property
    def cycle_fidelity_estimate(self) -> float:
        """Returns:
        Estimated cycle fidelity."""
        if self._cycle_fidelity_estimate is None:
            raise self._not_analyzed
        return self._cycle_fidelity_estimate

    @property
    def cycle_fidelity_estimate_std(self) -> float:
        """Returns:
        Standard deviation for the cycle fidelity estimate."""
        if self._cycle_fidelity_estimate_std is None:
            raise self._not_analyzed
        return self._cycle_fidelity_estimate_std

    def _analyze(self) -> None:
        """Analyse the results and calculate the estimated circuit fidelity.

        Args:
            plot_results (optional): Whether to generate the data plots. Defaults to True.

        Raises:
            RuntimeError: If there is no data stored.

        Returns:
           The final results from the experiment.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot perform analysis.")
        self.data["sum_p(x)p^(x)"] = pd.DataFrame(
            (
                self.data[["00", "01", "10", "11"]].values
                * self.data[["exact_00", "exact_01", "exact_10", "exact_11"]].values
            ).sum(axis=1),
            index=self.data.index,
        )
        self.data["sum_p(x)p(x)"] = pd.DataFrame(
            (
                self.data[["exact_00", "exact_01", "exact_10", "exact_11"]].values
                * self.data[["exact_00", "exact_01", "exact_10", "exact_11"]].values
            ).sum(axis=1),
            index=self.data.index,
        )
        # Fit a linear model for each cycle depth to estimate the circuit fidelity
        records = []
        for depth in set(self.data.cycle_depth):
            df = self.data[self.data.cycle_depth == depth]
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

        self._circuit_fidelities["log_fidelity_estimate"] = np.log(
            self._circuit_fidelities.circuit_fidelity_estimate
        )

        # Fit a linear model to the depth ~ log(fidelity) to approximate the cycle fidelity
        cycle_fit = scipy.stats.linregress(
            x=self._circuit_fidelities.cycle_depth,
            y=np.log(self._circuit_fidelities.circuit_fidelity_estimate),
        )
        self._cycle_fidelity_estimate = np.exp(cycle_fit.slope)
        self._cycle_fidelity_estimate_std = self.cycle_fidelity_estimate * cycle_fit.stderr

    def plot_results(self) -> None:
        """Plot the experiment data and the corresponding fits.

        Raises:
            RuntimeError: If there is no data stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot plot results.")

        if self._circuit_fidelities is None:
            raise RuntimeError(
                "No stored dataframe of circuit fidelities. Something has gone wrong."
            )

        plot_1 = sns.lmplot(
            data=self.data,
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
            data=self._circuit_fidelities,
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
            self._circuit_fidelities.cycle_depth.min(), self._circuit_fidelities.cycle_depth.max()
        )
        y = self.cycle_fidelity_estimate**x
        y_p = (self.cycle_fidelity_estimate + self.cycle_fidelity_estimate_std) ** x
        y_m = (self.cycle_fidelity_estimate - self.cycle_fidelity_estimate_std) ** x
        ax_2.plot(x, y, color="tab:red", linewidth=2)
        ax_2.fill_between(x, y_m, y_p, alpha=0.2, color="tab:red")

    def print_results(self) -> None:
        print(
            f"Estimated cycle fidelity: {self.cycle_fidelity_estimate:.5} "
            f"+/- {self.cycle_fidelity_estimate_std:.5}"
        )

    def plot_speckle(self) -> None:
        """Creates the speckle plot of the XEB data. See Fig. S18 of
        https://arxiv.org/abs/1910.11333 for an explanation of this plot.
        """
        df = self.data
        df2 = pd.melt(
            df,
            value_vars=["00", "01", "10", "11"],
            id_vars=["cycle_depth", "circuit_index"],
            var_name="bitstring",
        )
        fig, axs = plt.subplots(nrows=4, sharex=True)
        fig.subplots_adjust(hspace=0)
        for k, bitstring in enumerate(["00", "01", "10", "11"]):
            data = df2[df2["bitstring"] == bitstring].pivot(
                index="circuit_index", columns="cycle_depth", values="value"
            )
            cmap = mpl.colormaps["rocket"]
            norm = mpl.colors.Normalize(0, 1)  # or vmin, vmax
            sns.heatmap(data, vmin=0, vmax=1, ax=axs[k], cbar=False, cmap=cmap)
            axs[k].set_ylabel("")
            axs[k].set_xlabel("")
            axs[k].set_yticks([0, 15])
            axs[k].set_yticklabels([0, 15])
            plt.text(
                0.99,
                0.90,
                f"P({bitstring})",
                ha="right",
                va="top",
                transform=axs[k].transAxes,
                color="white",
            )
            if k != 0:
                axs[k].axhline(y=0, linewidth=1.5, color="white", linestyle="--")
        fig.supxlabel("Cycle depth")
        fig.supylabel("Circuit Instance")
        fig.colorbar(
            mpl.cm.ScalarMappable(norm, cmap), ax=axs, orientation="vertical", label="Probability"
        )


class XEB(QCVVExperiment[XEBResults]):
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
        num_circuits: int,
        cycle_depths: Iterable[int],
        single_qubit_gate_set: list[cirq.Gate] | None = None,
        two_qubit_gate: cirq.Gate | None = cirq.CZ,
        *,
        random_seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initializes a cross-entropy benchmarking experiment.

        Args:
            num_circuits: Number of circuits to sample.
            cycle_depths: The cycle depths to sample.
            single_qubit_gate_set: Optional list of single qubit gates to randomly sample from when
                generating random circuits. If not provided defaults to phased XZ gates with 1/4 pi
                intervals.
            two_qubit_gate: The two qubit gate to interleave between the single qubit gates. If None
                then no two qubit gate is used. Defaults to control-Z gate.
            random_seed: An optional seed to use for randomization.
        """
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

        super().__init__(
            num_qubits=2,
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            random_seed=random_seed,
            results_cls=XEBResults,
        )

    ###################
    # Private Methods #
    ###################
    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
    ) -> Sequence[Sample]:
        """Build a list of random circuits to perform the XEB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.

        Returns:
            The list of experiment samples.
        """
        random_circuits = []
        for k, depth in tqdm.contrib.itertools.product(
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

            analytic_final_state = cirq.final_state_vector(
                circuit, qubit_order=sorted(circuit.all_qubits())
            )
            analytic_probabilities = {
                "exact_" + format(idx, f"0{self.num_qubits}b"): np.abs(state) ** 2
                for idx, state in enumerate(analytic_final_state)
            }

            random_circuits.append(
                Sample(
                    circuit=circuit + cirq.measure(sorted(circuit.all_qubits())),
                    data={
                        "circuit_depth": len(circuit),
                        "cycle_depth": depth,
                        "two_qubit_gate": str(self.two_qubit_gate),
                        **analytic_probabilities,
                    },
                    circuit_index=k,
                )
            )

        return random_circuits
