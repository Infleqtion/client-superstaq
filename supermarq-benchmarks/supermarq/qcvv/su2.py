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
"""Tooling for SU(2) benchmarking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cirq
import cirq_superstaq as css
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
from tqdm.contrib.itertools import product

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample

if TYPE_CHECKING:
    from typing import Self


@dataclass
class SU2Results(QCVVResults):
    """Data structure for the SU2 experiment results."""

    _two_qubit_gate_fidelity: float | None = None
    """Estimated two qubit gate fidelity"""
    _two_qubit_gate_fidelity_std: float | None = None
    """Standard deviation of the two qubit gate fidelity estimate"""
    _single_qubit_noise: float | None = None
    """Estimated single qubit noise."""
    _single_qubit_noise_std: float | None = None
    """Standard deviation of the single qubit noise estimate"""

    @property
    def two_qubit_gate_fidelity(self) -> float:
        """Returns:
        Estimated two qubit gate fidelity."""
        if self._two_qubit_gate_fidelity is None:
            raise self._not_analyzed
        return self._two_qubit_gate_fidelity

    @property
    def two_qubit_gate_fidelity_std(self) -> float:
        """Returns:
        Standard deviation of estimated two qubit gate fidelity."""
        if self._two_qubit_gate_fidelity_std is None:
            raise self._not_analyzed
        return self._two_qubit_gate_fidelity_std

    @property
    def single_qubit_noise(self) -> float:
        """Returns:
        Estimated single qubit noise."""
        if self._single_qubit_noise is None:
            raise self._not_analyzed
        return self._single_qubit_noise

    @property
    def single_qubit_noise_std(self) -> float:
        """Returns:
        Standard deviation of estimated single qubit noise."""
        if self._single_qubit_noise_std is None:
            raise self._not_analyzed
        return self._single_qubit_noise_std

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

    def plot_results(self, filename: str | None = None) -> plt.Figure:
        """Plot the results of the experiment.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A single matplotlib figure containing the relevant plots of the results data.

        Raises:
            RuntimeError: If there is no data stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot plot results.")

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=self.data.drop(columns=["circuit_realization", "uuid"]).melt(
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
            xx := self.data["num_two_qubit_gates"],
            3 / 4 * (1 - self.single_qubit_noise) * self.two_qubit_gate_fidelity**xx + 0.25,
            label="00 (fit)",
        )
        ax.set_xlabel("Number of two qubit gates")
        ax.set_ylabel("State probability")
        ax.legend(title="State")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)

        return fig

    def _analyze(self) -> None:
        """Perform the experiment analysis and store the results in the `results` attribute.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot perform analysis.")

        fit = linregress(
            x=self.data["num_two_qubit_gates"],
            y=np.log(4 / 3 * (self.data["00"] - 1 / 4)),
            # 1/4 < self.data["00"] < 1 so we subtract 1/4 and rescale by 4/3 to obtain a
            # quantity in the range 0 < 4 / 3 * (self.data["00"] - 1 / 4) < 1
        )
        gate_fid = np.exp(fit.slope)
        gate_fid_std = fit.stderr * gate_fid

        single_qubit_noise = 1 - np.exp(fit.intercept)
        single_qubit_noise_std = fit.intercept_stderr * (1 - single_qubit_noise)

        # Save results
        self._two_qubit_gate_fidelity = gate_fid
        self._two_qubit_gate_fidelity_std = gate_fid_std
        self._single_qubit_noise = single_qubit_noise
        self._single_qubit_noise_std = single_qubit_noise_std

    def print_results(self) -> None:
        """Prints the key results data."""
        print(  # noqa: T201
            f"Estimated two qubit gate fidelity: {self.two_qubit_gate_fidelity:.5} "
            f"+/- {self.two_qubit_gate_error_std:.5}\n"
            f"Estimated single qubit noise: {self.single_qubit_noise:.5} "
            f"+/- {self.single_qubit_noise_std:.5}\n"
        )


class SU2(QCVVExperiment[SU2Results]):
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

    See Fig. 3 of https://www.nature.com/articles/s41586-023-06481-y#Fig3 for further
    details.
    """

    def __init__(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
        two_qubit_gate: cirq.Gate = cirq.CZ,
        *,
        random_seed: int | np.random.Generator | None = None,
        _samples: list[Sample] | None = None,
        **kwargs: str,
    ) -> None:
        """Args:
        two_qubit_gate: The Clifford gate to measure the gate error of.
        num_qubits: The number of qubits to experiment on. Must equal 2.
        """
        if two_qubit_gate.num_qubits() != 2:
            raise ValueError(
                "The `two_qubit_gate` parameter must be a gate that acts on exactly two qubits."
            )
        self.two_qubit_gate = two_qubit_gate
        """The two qubit gate to be benchmarked"""

        super().__init__(
            qubits=2,
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            random_seed=random_seed,
            results_cls=SU2Results,
            _samples=_samples,
            **kwargs,
        )

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
        for depth, index in product(cycle_depths, range(num_circuits), desc="Building circuits"):
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

            samples.append(
                Sample(
                    circuit_realization=index,
                    circuit=circuit,
                    data={"num_two_qubit_gates": 2 * depth},
                )
            )
        return samples

    @staticmethod
    def _haar_random_rotation() -> cirq.Gate:
        """Returns:
        Haar randomly sampled SU(2) rotation.
        """
        gate = cirq.single_qubit_matrix_to_phxz(cirq.testing.random_special_unitary(dim=2))
        if gate is None:
            return cirq.I
        return gate

    def _component(self, include_two_qubit_gate: bool) -> cirq.Circuit:
        """Core component of the experimental circuits.

        These circuits that are repeated to create the full circuit. Can optionally include the
        two qubit gate being measured, as is required for the
        first half of the full circuit, but not for the second half.

        The component looks like:
        .. code::

            0: ───R───Q───X───Q───
                      │       │
            1: ───R───Q───X───Q───

        where :code:`R` is a Haar randomly chosen SU(2) rotation
        and :code:`Q-Q` represents the two qubit gate being measured.

        Args:
            include_two_qubit_gate: Whether to include the two qubit gate being measured

        Returns:
            The sub circuit to be repeated when building the full circuit
        """
        # Chose a global rotation
        rotation = self._haar_random_rotation()
        return cirq.Circuit(
            rotation.on(self.qubits[0]),
            rotation.on(self.qubits[1]),
            css.barrier(*self.qubits),
            (
                [
                    self.two_qubit_gate(*self.qubits).with_tags("no_compile"),
                    css.barrier(*self.qubits),
                ]
                if include_two_qubit_gate
                else []
            ),
            cirq.X.on_each(*self.qubits),
            css.barrier(*self.qubits),
            (
                [
                    self.two_qubit_gate(*self.qubits).with_tags("no_compile"),
                    css.barrier(*self.qubits),
                ]
                if include_two_qubit_gate
                else []
            ),
        )

    def _json_dict_(self) -> dict[str, Any]:
        """Converts the experiment to a json-able dictionary that can be used to recreate the
        experiment object. Note that the state of the random number generator is not stored.

        Returns:
            Json-able dictionary of the experiment data.
        """
        return {
            "two_qubit_gate": self.two_qubit_gate,
            **super()._json_dict_(),
        }

    @classmethod
    def _from_json_dict_(
        cls,
        _samples: list[Sample],
        num_circuits: int,
        cycle_depths: Iterable[int],
        two_qubit_gate: cirq.Gate = cirq.CZ,
        **kwargs: Any,
    ) -> Self:
        """Creates a experiment from a dictionary of the data.

        Args:
            dictionary: Dict containing the experiment data.

        Returns:
            The deserialized experiment object.
        """
        kwargs.pop("qubits")  # Number of qubits is fixed for SU2
        return cls(
            num_circuits=num_circuits,
            _samples=_samples,
            cycle_depths=cycle_depths,
            two_qubit_gate=two_qubit_gate,
            **kwargs,
        )
