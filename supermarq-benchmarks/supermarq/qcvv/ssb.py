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
"""Tooling for Symmetric Stabilizer Benchmarking.
See https://arxiv.org/pdf/2407.20184 for more details.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cirq
import cirq_superstaq as css
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
import seaborn as sns
import tqdm
import tqdm.contrib
import tqdm.contrib.itertools

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample

if TYPE_CHECKING:
    from typing_extensions import Self


def _exp_decay(
    x: npt.NDArray[np.float64], A: float, alpha: float, B: float
) -> npt.NDArray[np.float64]:
    r"""Exponential decay of the form.

    .. math::

        A \alpha^x + B

    Args:
        x: x
        A: Decay constant
        alpha: Decay coefficient
        B: Additive constant
    Returns:
        Exponential decay applied to x.
    """
    return A * (alpha**x) + B


@dataclass
class SSBResults(QCVVResults):
    """Results from an SSB experiment."""

    _cz_fidelity_estimate: float | None = None
    """Estimated CZ fidelity."""
    _cz_fidelity_estimate_std: float | None = None
    """Standard deviation for the CZ fidelity estimate."""
    _fit: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
    """The fitted values"""

    @property
    def cz_fidelity_estimate(self) -> float:
        """Estimated CZ fidelity."""
        if self._cz_fidelity_estimate is None:
            raise self._not_analyzed
        return self._cz_fidelity_estimate

    @property
    def cz_fidelity_estimate_std(self) -> float:
        """Standard deviation for the CZ fidelity estimate."""
        if self._cz_fidelity_estimate_std is None:
            raise self._not_analyzed
        return self._cz_fidelity_estimate_std

    def _analyze(self) -> None:
        """Analyse the results and calculate the estimated CZ gate fidelity.

        Args:
            plot_results (optional): Whether to generate the data plots. Defaults to True.

        Returns:
           The final results from the experiment.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot perform analysis.")

        xx = self.data["num_cz_gates"]
        yy = self.data["0" * self.num_qubits]

        self._fit = scipy.optimize.curve_fit(
            _exp_decay,
            xx,
            yy,
            p0=(1.0 - 2**-self.num_qubits, 0.99, 2**-self.num_qubits),
            bounds=(0, 1),
            max_nfev=2000,
        )
        assert self._fit is not None

        self._cz_fidelity_estimate = self._fit[0][1]
        self._cz_fidelity_estimate_std = np.sqrt(self._fit[1][1][1])

    def plot_results(self, filename: str | None = None) -> plt.Figure:
        """Plot the experiment data and the corresponding fits. The shaded upper and lower limits
        of the shaded region indicate the fits at +/- 1 standard deviation in all fitted parameters.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A single matplotlib figure with the experimental data and corresponding fits.

        Raises:
            RuntimeError: If there is no data stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot plot results.")
        assert self._fit is not None

        fig, axs = plt.subplots(1, 1)
        sns.scatterplot(
            data=self.data, x="num_cz_gates", y="0" * self.num_qubits, ax=axs, label="Data"
        )
        xx = np.linspace(0, np.max(self.data.num_cz_gates))
        axs.plot(xx, _exp_decay(xx, *self._fit[0]), color="tab:blue", linestyle="--", label="Fit")
        axs.fill_between(
            xx,
            _exp_decay(xx, *(self._fit[0] - np.sqrt(np.diag(self._fit[1])))),
            _exp_decay(xx, *(self._fit[0] + np.sqrt(np.diag(self._fit[1])))),
            alpha=0.35,
            color="tab:blue",
            label="1s.d C.I.",
        )
        axs.legend()

        axs.set_xlabel("Number of CZ Gates")
        axs.set_ylabel("Final state probability")

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        return fig

    def print_results(self) -> None:
        """Prints the key results data."""
        print(  # noqa: T201
            f"Estimated CZ fidelity: {self.cz_fidelity_estimate:.5} "
            f"+/- {self.cz_fidelity_estimate_std:.5}"
        )


class SSB(QCVVExperiment[SSBResults]):
    """Symmetric Stabilizer Benchmarking. A benchmarking algorithm for determining the CZ fidelity
    of a device. Specifically designed for neutral atom devices where CZ-gates mediated by Rydberg
    interactions are the native entangling gate.

    See: https://arxiv.org/abs/2407.20184
    """

    def __init__(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
        *,
        include_placeholders: bool = False,
        distribute: bool = True,
        random_seed: int | np.random.Generator | None = None,
        _samples: list[Sample] | None = None,
        **kwargs: str,
    ) -> None:
        """Initializes a symmetric stabilizer benchmarking experiment.

        Args:
            num_circuits: Number of circuits to sample.
            cycle_depths: The cycle depths to sample.
            random_seed: An optional seed to use for randomization.
            include_placeholders: Whether to include placeholder barriers to prevent adjacent layer
                being compiled together.
            distribute: Whether to randomly distribute the CZ gates throughout the the circuit
                instead of having them all at the start.
            kwargs: Additional kwargs passed to the base QCVVExperiment.
        """
        qubits = cirq.LineQubit.range(2)

        self._include_placeholders = include_placeholders
        self._distribute = distribute

        # Moments containing parallel rotations.
        X = css.ParallelRGate(np.pi / 2, 0.0, 2)
        Y = css.ParallelRGate(np.pi / 2, np.pi / 2, 2)
        _X = css.ParallelRGate(np.pi / 2, np.pi, 2)
        _Y = css.ParallelRGate(np.pi / 2, -np.pi / 2, 2)
        self._single_qubit_gate_set = [X, _X, Y, _Y]

        # Table I of https://arxiv.org/pdf/2407.20184
        stabilizer_states = np.array(
            [
                [1, 1, 1, 1],
                [1, -1, -1, 1],
                [1, 1j, 1j, -1],
                [1, -1j, -1j, -1],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 1, 1, -1],
                [1, -1, -1, -1],
                [1, 1j, 1j, 1],
                [1, -1j, -1j, 1],
                [1, 0, 0, 1j],
                [1, 0, 0, -1j],
            ]
        )
        stabilizer_states /= np.linalg.norm(stabilizer_states, axis=1, keepdims=True)
        self._stabilizer_states = stabilizer_states

        # Table II of https://arxiv.org/pdf/2407.20184
        self._init_rotations = [
            [_X, X, Y, _Y, Y],
            [_X, _X, Y, _Y, Y],
            [_X, _X, X, _X, X],
            [_X, X, X, _X, X],
            [_X, X, X, _Y, _X],
            [_X, _X, X, _Y, _X],
            [X, _Y, X, _Y, _X],
            [_X, Y, X, _Y, _X],
            [X, _Y, X, _X, _X],
            [_X, Y, X, _X, X],
            [_X, Y, Y, _Y, Y],
            [X, _Y, Y, _Y, Y],
        ]
        # Table III of https://arxiv.org/pdf/2407.20184
        self._reconciliation_rotation = [
            [X, _Y, X, _X],
            [X, Y, X, _X],
            [Y, X, X, _X],
            [Y, _X, X, _X],
            [_X, X, X, _X],
            [X, X, X, _X],
            [_X, Y, Y, _X],
            [X, Y, Y, _X],
            [Y, Y, Y, _X],
            [X, X, Y, _X],
            [_Y, X, Y, _X],
            [Y, X, Y, _X],
        ]

        super().__init__(
            qubits=qubits,
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            random_seed=random_seed,
            results_cls=SSBResults,
            _samples=_samples,
            **kwargs,
        )

    ###################
    # Private Methods #
    ###################
    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: Iterable[int],
    ) -> Sequence[Sample]:
        """Build a list of random circuits to perform the SSB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.

        Returns:
            The list of experiment samples.
        """
        random_circuits = []
        min_depth = min(cycle_depths)
        max_depth = max(cycle_depths)
        if min_depth < 2:
            raise ValueError("Cannot perform SSB with a cycle depth of 1.")

        # Precompute action of each gate on the stabilizer state index. This is just Clifford
        # tableau math but we only have to do this once though so there really isn't much of a
        # performance difference
        idx_maps = {}
        for gate in (*self._single_qubit_gate_set, cirq.CZ):
            mat = cirq.unitary(gate)
            inner_products = abs(self._stabilizer_states.conj() @ mat @ self._stabilizer_states.T)
            idx_maps[gate] = dict(enumerate(inner_products.argmax(0)))

        # Build samples
        for k, depth in tqdm.contrib.itertools.product(
            range(num_circuits), cycle_depths, desc="Building circuits"
        ):
            sss_idx = int(self._rng.integers(0, 11, endpoint=True))
            circuit = self._sss_init_circuit(sss_idx)

            if self._distribute and depth < max_depth:
                indices = self._rng.choice(max_depth - 2, depth - 2, replace=False)
            else:
                indices = np.arange(depth - 2)

            num = 0
            for i in range(max_depth - 2):
                if i in indices:
                    circuit += cirq.CZ.on(*self.qubits).with_tags("no_compile")
                    sss_idx = idx_maps[cirq.CZ][sss_idx]
                    num += 1

                if self._include_placeholders:
                    circuit += css.Barrier(2)(*self.qubits)

                gate = self._random_parallel_qubit_rotation()
                circuit += gate.on(*self.qubits)
                sss_idx = idx_maps[gate][sss_idx]

            assert num == depth - 2, (max_depth, depth, num)

            circuit += self._sss_reconciliation_circuit(sss_idx) + cirq.measure(self.qubits)

            random_circuits.append(
                Sample(
                    circuit=circuit,
                    data={
                        "num_cz_gates": depth,
                        "initial_sss_index": sss_idx,
                    },
                    circuit_realization=k,
                )
            )
        return random_circuits

    def _random_parallel_qubit_rotation(self) -> cirq.Gate:
        """Chooses randomly from {X, Y, -X, -Y} and return a moment with this gate acting on both
        quits. Note we don't use `cirq.ParallelGate` as when modelling noise Cirq treats this as
        a two qubit gate.

        Returns:
            The randomly chosen rotation gate acting on both qubits.
        """
        return self._rng.choice(np.array(self._single_qubit_gate_set))

    def _sss_init_circuit(self, idx: int) -> cirq.Circuit:
        """Creates the initialisation circuit for the provided symmetric-stabiliser state index.
        See appendix of https://arxiv.org/pdf/2407.20184 for details.

        Args:
            idx: The index of the desired symmetric-stabiliser state

        Returns:
            A circuit that maps the |00> state to the desired symmetric-stabiliser state.
        """
        init_circuit = cirq.Circuit(
            self._init_rotations[idx][0].on(*self.qubits),
            self._init_rotations[idx][1].on(*self.qubits),
            cirq.CZ(*self.qubits).with_tags("no_compile"),
            self._init_rotations[idx][2].on(*self.qubits),
            self._init_rotations[idx][3].on(*self.qubits),
            self._init_rotations[idx][4].on(*self.qubits),
        )
        return init_circuit

    def _sss_reconciliation_circuit(self, idx: int) -> cirq.Circuit:
        """Given a randomly generated circuit, return the appropriate reconciliation circuit.
        See appendix of https://arxiv.org/pdf/2407.20184 for details.

        Args:
            idx: The index of the final symmetric-stabiliser state

        Returns:
            The appropriate reconciliation circuit
        """
        # Return the reconciliation circuit. See table III of https://arxiv.org/pdf/2407.20184
        return cirq.Circuit(
            self._reconciliation_rotation[idx][0].on(*self.qubits),
            self._reconciliation_rotation[idx][1].on(*self.qubits),
            cirq.CZ(*self.qubits).with_tags("no_compile"),
            self._reconciliation_rotation[idx][2].on(*self.qubits),
            self._reconciliation_rotation[idx][3].on(*self.qubits),
        )

    def _json_dict_(self) -> dict[str, Any]:
        """Converts the experiment to a json-able dictionary that can be used to recreate the
        experiment object. Note that the state of the random number generator is not stored.

        Returns:
            Json-able dictionary of the experiment data.
        """
        return super()._json_dict_()

    @classmethod
    def _from_json_dict_(
        cls,
        _samples: list[Sample],
        num_circuits: int,
        cycle_depths: list[int],
        **kwargs: Any,
    ) -> Self:
        """Creates a experiment from a dictionary of the data.

        Args:
            dictionary: Dict containing the experiment data.

        Returns:
            The deserialized experiment object.
        """
        kwargs.pop("qubits")  # Don't need for SSB

        return cls(
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            _samples=_samples,
            **kwargs,
        )
