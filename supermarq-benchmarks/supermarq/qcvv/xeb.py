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
"""Tooling for cross entropy benchmark experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cirq
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm.contrib.itertools

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample

if TYPE_CHECKING:
    import numpy.typing as npt


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
        """Estimated cycle fidelity."""
        if self._cycle_fidelity_estimate is None:
            raise self._not_analyzed
        return self._cycle_fidelity_estimate

    @property
    def cycle_fidelity_estimate_std(self) -> float:
        """Standard deviation for the cycle fidelity estimate."""
        if self._cycle_fidelity_estimate_std is None:
            raise self._not_analyzed
        return self._cycle_fidelity_estimate_std

    def _analytical_data(self) -> pd.DataFrame:
        """Create a copy of `self.data` with analytical probabilities in place of experimental data.

        Returns:
            The `pd.DataFrame` containing analytical probabilities.
        """
        assert self.data is not None

        indices = [f"{i:0>{self.num_qubits}b}" for i in range(2**self.num_qubits)]

        analytic_probabilities: list[npt.NDArray[np.float64]] = []
        for _, row in self.data.iterrows():
            sample = self.experiment[row.uuid]
            subcircuit = sample.circuit.copy()
            subcircuit.clear_operations_touching(
                subcircuit.all_qubits().difference(self.qubits), range(len(subcircuit))
            )
            analytic_final_state = cirq.final_state_vector(
                subcircuit,
                qubit_order=self.qubits,
                ignore_terminal_measurements=True,
                dtype=np.complex128,
            )
            analytic_probabilities.append(np.abs(analytic_final_state) ** 2)

        analytical_data = self.data.drop(columns=indices, errors="ignore")
        analytical_data[indices] = analytic_probabilities
        return analytical_data

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

        analytical_data = self._analytical_data()

        indices = [f"{i:0>{self.num_qubits}b}" for i in range(2**self.num_qubits)]
        self.data["sum_p(x)p^(x)"] = (self.data[indices] * analytical_data[indices]).sum(axis=1)
        self.data["sum_p(x)p(x)"] = (analytical_data[indices] ** 2).sum(axis=1)

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
        self._zero_cycle_fidelity_estimate = np.exp(cycle_fit.intercept)

    def plot_results(
        self,
        filename: str | None = None,
    ) -> plt.Figure:
        """Plot the experiment data and the corresponding fits.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A single matplotlib figure containing both the linear fit per cycle depth and the decay
            with cycle depth.

        Raises:
            RuntimeError: If there is no data stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot plot results.")

        if self._circuit_fidelities is None:
            raise RuntimeError(
                "No stored dataframe of circuit fidelities. Something has gone wrong."
            )

        fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
        colours = sns.color_palette("dark:r", n_colors=len(self.data.cycle_depth.unique()))
        for depth, color in zip(sorted(self.data.cycle_depth.unique()), colours):
            sns.regplot(
                data=self.data[self.data.cycle_depth == depth],
                x="sum_p(x)p(x)",
                y="sum_p(x)p^(x)",
                ci=None,
                ax=axs[0],
                color=color,
                label=depth,
            )
        axs[0].legend(title="Cycle depth", bbox_to_anchor=(1.0, 0.5), loc="center left")
        axs[0].set_xlabel(r"$\sum p(x)^2$", fontsize=15)
        axs[0].set_ylabel(r"$\sum p(x) \hat{p}(x)$", fontsize=15)
        axs[0].set_title(r"Linear fit per cycle depth", fontsize=15, wrap=True)

        sns.regplot(
            data=self._circuit_fidelities,
            x="cycle_depth",
            y="circuit_fidelity_estimate",
            ax=axs[1],
            fit_reg=False,
            color="tab:red",
        )
        axs[1].set_xlabel(r"Cycle depth", fontsize=15)
        axs[1].set_ylabel(r"Circuit fidelity", fontsize=15)
        axs[1].set_title(r"Exponential decay of circuit fidelity", fontsize=15, wrap=True)

        # Add fit line
        x = np.linspace(
            self._circuit_fidelities.cycle_depth.min(), self._circuit_fidelities.cycle_depth.max()
        )
        y = self.cycle_fidelity_estimate**x * self._zero_cycle_fidelity_estimate
        y_p = (
            self.cycle_fidelity_estimate + 1.96 * self.cycle_fidelity_estimate_std
        ) ** x * self._zero_cycle_fidelity_estimate
        y_m = (
            self.cycle_fidelity_estimate - 1.96 * self.cycle_fidelity_estimate_std
        ) ** x * self._zero_cycle_fidelity_estimate
        axs[1].plot(x, y, color="tab:red", linewidth=2)
        axs[1].fill_between(x, y_m, y_p, alpha=0.25, color="tab:red", label="95% CI")
        axs[1].legend()

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        return fig

    def print_results(self) -> None:
        """Prints the key results data."""
        print(  # noqa: T201
            f"Estimated cycle fidelity: {self.cycle_fidelity_estimate:.5} "
            f"+/- {self.cycle_fidelity_estimate_std:.5}"
        )

    def plot_speckle(self, filename: str | None = None) -> plt.Figure:
        """Creates the speckle plot of the XEB data. See Fig. S18 of
        https://arxiv.org/abs/1910.11333 for an explanation of this plot.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A matplotlib figure with the speckle plot.

        Raises:
            RuntimeError: If there is no data stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot plot results.")

        indices = [f"{i:0>{self.num_qubits}b}" for i in range(2**self.num_qubits)]

        # Reformat dataframe
        df = pd.melt(
            self.data,
            value_vars=indices,
            id_vars=["cycle_depth", "circuit_realization"],
            var_name="bitstring",
        )

        # Create the axes needed
        fig, axs = plt.subplot_mosaic(
            [[f"P({idx})", "cbar", ".", "Decay"] for idx in indices],
            width_ratios=[1, 0.05, 0.05, 1],
            figsize=(12, 4.8),
        )
        fig.subplots_adjust(hspace=0)

        # Plot the heatmaps
        for k, bitstring in enumerate(indices):
            ax = axs[f"P({bitstring})"]

            data = df[df["bitstring"] == bitstring].pivot(
                index="circuit_realization", columns="cycle_depth", values="value"
            )
            cmap = mpl.colormaps["rocket"]
            # norm = mpl.colors.Normalize(0, 1)  # or vmin, vmax # noqa: ERA001
            sns.heatmap(data, vmin=0, vmax=1, ax=ax, cbar_ax=axs["cbar"], cmap=cmap)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.set_yticklabels([])
            plt.text(
                0.99,
                0.90,
                f"P({bitstring})",
                ha="right",
                va="top",
                transform=ax.transAxes,
                color="white",
            )
            if k == 0:
                ax.set_title("Speckle plots")
            else:
                ax.axhline(y=0, linewidth=1.5, color="white", linestyle="--")

            if k == 2**self.num_qubits - 1:
                ax.set_xlabel("Cycle depth")

        # Format colour bar
        axs["cbar"].set_ylabel("Probability")
        axs["cbar"].yaxis.set_label_position("right")
        axs["cbar"].yaxis.tick_left()

        # Plot the decay of purity

        # Calculate the average std of the probability distributions.
        purity_data = (
            df.groupby(by=["cycle_depth"])
            .std(numeric_only=True)
            .reset_index()
            .rename(columns={"value": "sqrt_speckle_purity"})
            .drop(columns=["circuit_realization"])
        )
        # Rescale the purity estimate according to Porter-Thomas distribution
        dim = np.prod([q.dimension for q in self.qubits])
        purity_data["sqrt_speckle_purity"] = purity_data["sqrt_speckle_purity"] * np.sqrt(
            dim**2 * (dim + 1) / (dim - 1)
        )

        # Plot decay
        sns.regplot(
            data=purity_data,
            x="cycle_depth",
            y="sqrt_speckle_purity",
            logx=True,
            ax=axs["Decay"],
        )
        # purity_plot.tight_layout() # noqa: ERA001
        axs["Decay"].set_xlabel(r"Cycle depth")
        axs["Decay"].set_ylabel(r"$\sqrt{\mathrm{Purity}}$")
        axs["Decay"].set_title(r"Purity Decay")
        # Estimate decay coefficient
        purity_fit = scipy.stats.linregress(
            x=purity_data["cycle_depth"],
            y=np.log(purity_data["sqrt_speckle_purity"]),
        )
        # Add label with coefficient estimate
        axs["Decay"].text(
            0.95,
            0.95,
            (
                f"Decay coefficient: {np.exp(purity_fit.slope):4f} "
                f"+/- {np.exp(purity_fit.slope) * purity_fit.stderr:4f}"
            ),
            ha="right",
            va="center",
            transform=axs["Decay"].transAxes,
        )

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        return fig


class XEB(QCVVExperiment[XEBResults]):
    r"""Cross-entropy benchmarking (XEB) experiment.

    The XEB experiment can be used to estimate the combined fidelity of a repeating cycle of gates.
    Cycles are made up of randomly selected single qubit gates and a constant layer of interest.
    This is illustrated as follows:

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
        interleaved_layer: cirq.Gate | cirq.OP_TREE | None = cirq.CZ,
        single_qubit_gate_set: Sequence[cirq.Gate] | None = None,
        *,
        random_seed: int | np.random.Generator | None = None,
        _samples: list[Sample] | None = None,
        **kwargs: str,
    ) -> None:
        """Initializes a cross-entropy benchmarking experiment.

        Args:
            num_circuits: Number of circuits to sample.
            cycle_depths: The cycle depths to sample.
            interleaved_layer: The gate or operation(s) to interleave between the single qubit
                gates. If None then no gates are interleaved.
            single_qubit_gate_set: Optional list of single qubit gates to randomly sample from when
                generating random circuits. If not provided defaults to phased X gates with 1/4 pi
                intervals.
            random_seed: An optional seed to use for randomization.
            kwargs: Any additional supported string keyword args.
        """
        if interleaved_layer is None:
            qubits: Sequence[cirq.Qid] = cirq.LineQubit.range(2)

        elif isinstance(interleaved_layer, cirq.Gate):
            qubits = (
                cirq.LineQubit.range(cirq.num_qubits(interleaved_layer))
                if all(d == 2 for d in cirq.qid_shape(interleaved_layer))
                else cirq.LineQid.for_gate(interleaved_layer)
            )
            interleaved_layer = interleaved_layer.on(*qubits)

        elif isinstance(interleaved_layer, (cirq.Operation, cirq.Moment)):
            qubits = sorted(interleaved_layer.qubits)

        else:
            interleaved_layer = cirq.Circuit(interleaved_layer)
            qubits = sorted(interleaved_layer.all_qubits())

        self.interleaved_layer: cirq.OP_TREE | None = interleaved_layer
        """The layer to interleave."""

        self.single_qubit_gate_set: list[cirq.Gate]
        """The single qubit gates to randomly sample from"""

        if single_qubit_gate_set is None:
            self.single_qubit_gate_set = [
                cirq.PhasedXPowGate(exponent=0.5, phase_exponent=phase_exponent)
                for phase_exponent in [0.0, 0.25, 0.5]
            ]
        else:
            self.single_qubit_gate_set = list(single_qubit_gate_set)

        super().__init__(
            qubits=qubits,
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            random_seed=random_seed,
            results_cls=XEBResults,
            _samples=_samples,
            **kwargs,
        )

    def independent_qubit_groups(self) -> list[tuple[cirq.Qid, ...]]:
        """Get all independent subsets of qubits in this experiment.

        Returns:
            A list of disjoint tuples of `cirq.Qid` objects, each of which can be analyzed as an
            independent XEB experiment.
        """
        if self.interleaved_layer:
            qubit_sets = cirq.Circuit(self.interleaved_layer).get_independent_qubit_sets()
            return sorted((tuple(sorted(qubit_set)) for qubit_set in qubit_sets), key=min)

        return [self.qubits]

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
            # Choose single-qubit gates, avoiding repeats on the same qubit in sequential layers
            num_choices = len(self.single_qubit_gate_set)
            block_repeats = num_choices > 1
            chosen_gate_indices = np.append(
                self._rng.integers(0, num_choices, size=(1, self.num_qubits)),
                self._rng.integers(block_repeats, num_choices, size=(depth, self.num_qubits)),
                axis=0,
            )
            chosen_gate_indices = np.add.accumulate(chosen_gate_indices, axis=0) % num_choices

            circuit = cirq.Circuit(
                self.single_qubit_gate_set[gate_index].on(qubit)
                for gate_indices_in_layer in chosen_gate_indices
                for gate_index, qubit in zip(gate_indices_in_layer, self.qubits)
            )

            circuit = self._interleave_layer(circuit, self.interleaved_layer)

            random_circuits.append(
                Sample(
                    circuit=circuit + cirq.measure(*self.qubits),
                    data={
                        "circuit_depth": self._count_non_barrier_gates(circuit),
                        "cycle_depth": depth,
                        "interleaved_layer": str(self.interleaved_layer),
                    },
                    circuit_realization=k,
                )
            )

        return random_circuits

    def _json_dict_(self) -> dict[str, Any]:
        """Converts the experiment to a json-able dictionary that can be used to recreate the
        experiment object. Note that the state of the random number generator is not stored.

        Returns:
            Json-able dictionary of the experiment data.
        """
        json_dict = super()._json_dict_()
        json_dict["interleaved_layer"] = self.interleaved_layer
        json_dict["single_qubit_gate_set"] = self.single_qubit_gate_set
        del json_dict["qubits"]

        return json_dict
