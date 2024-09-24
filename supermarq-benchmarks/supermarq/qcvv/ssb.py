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
See https://arxiv.org/pdf/2407.20184
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tqdm.contrib.itertools

from supermarq.qcvv import BenchmarkingExperiment, BenchmarkingResults, Sample


def _parallel(
    x: np.typing.NDArray[np.float_], y: np.typing.NDArray[np.float_], tol: float = 1e-5
) -> bool:
    """Checks whether two numpy arrays are parallel, as determined by the cosine distance between
    them.

    Args:
        x: Array 1
        y: Array 2
        tol: The tolerance to accept. Defaults to 1E-5.

    Returns:
        Wether the two arrays are parallel
    """
    return np.abs(np.dot(x, np.conj(y)) / (np.linalg.norm(x) * np.linalg.norm(y))) >= 1.0 - tol


@dataclass(frozen=True)
class SSBResults(BenchmarkingResults):
    """Results from an SSB experiment."""

    cz_fidelity_estimate: float
    """Estimated cycle fidelity."""
    cz_fidelity_estimate_std: float
    """Standard deviation for the cycle fidelity estimate."""

    experiment_name = "SSB"


class SSB(BenchmarkingExperiment[SSBResults]):
    """Symmetric Stabilizer Benchmarking"""

    def __init__(self) -> None:
        """Initializes a symmetric stabilizer benchmarking experiment."""

        super().__init__(num_qubits=2)

        # Moments containing parallel rotations. Used to construst the init and rec circuit.
        # Note we avoid using the `cirq.ParallelGate` as this can lead to single qubit gates
        # looking like two qubit gates when building custom noise models.
        X = css.ParallelRGate(np.pi / 2, 0.0, self.num_qubits)
        Y = css.ParallelRGate(np.pi / 2, np.pi / 2, self.num_qubits)
        _X = css.ParallelRGate(np.pi / 2, np.pi, self.num_qubits)
        _Y = css.ParallelRGate(np.pi / 2, -np.pi / 2, self.num_qubits)
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
        assert min_depth >= 2

        for _, depth in tqdm.contrib.itertools.product(
            range(num_circuits), cycle_depths, desc="Building circuits"
        ):
            sss_idx = random.randint(0, 11)
            circuit = self._sss_init_circuit(sss_idx)
            for _ in range(depth - 2):
                circuit += self._random_parallel_qubit_rotation()
                circuit += cirq.CZ(*self.qubits).with_tags("no_compile")
            for _ in range(max_depth - depth + 2):
                circuit += self._random_parallel_qubit_rotation()
            circuit += self._sss_reconciliation_circuit(circuit) + cirq.measure(self.qubits)

            random_circuits.append(
                Sample(
                    raw_circuit=circuit,
                    data={
                        "num_cz_gates": depth,
                        "initial_sss_index": sss_idx,
                    },
                )
            )
        return random_circuits

    def _process_probabilities(self, samples: Sequence[Sample]) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into the data structures
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
                    **sample.data,
                    **sample.probabilities,
                }
            )
        return pd.DataFrame(records)

    def _random_parallel_qubit_rotation(self) -> cirq.Operation:
        """Chooses randomly from {X, Y, -X, -Y} and return a moment with this gate acting on both
        quits. Note we don't use `cirq.ParallelGate` as when modelling noise Cirq treats this as
        a two qubit gate.

        Returns:
            The randomly chosen rotation gate acting on both qubits.
        """
        gate = random.choice(self._single_qubit_gate_set)
        return gate.on(*self.qubits)

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
            cirq.CZ(*self.qubits),
            self._init_rotations[idx][2].on(*self.qubits),
            self._init_rotations[idx][3].on(*self.qubits),
            self._init_rotations[idx][4].on(*self.qubits),
        )
        return init_circuit

    def _sss_reconciliation_circuit(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Given a randomly generated circuit, return the appropriate reconciliation circuit.
        See appendix of https://arxiv.org/pdf/2407.20184 for details.

        Args:
            circuit: The randomly generated circuit

        Returns:
            The appropriate reconciliation circuit
        """
        # Calculate which state the provided circuit maps the |00> state to.
        state = cirq.final_state_vector(circuit, ignore_terminal_measurements=True)

        # Find the index of this state by checking which symmetric-stabiliser
        # state it is parallel to.
        idx = abs(self._stabilizer_states @ state.conj()).argmax()

        # Return the reconciliation circuit. See table III of https://arxiv.org/pdf/2407.20184
        return cirq.Circuit(
            self._reconciliation_rotation[idx][0].on(*self.qubits),
            self._reconciliation_rotation[idx][1].on(*self.qubits),
            cirq.CZ(*self.qubits),
            self._reconciliation_rotation[idx][2].on(*self.qubits),
            self._reconciliation_rotation[idx][3].on(*self.qubits),
        )

    def _fit_decay(self) -> tuple[np.typing.NDArray[np.float_], np.typing.NDArray[np.float_]]:
        """Fits the exponential decay function to the survivial probability as a function of
        the number of CZ gates.

        Returns:
            Tuple of the decay fit parameters and their standard deviations.
        """

        xx = self.raw_data["num_cz_gates"]
        yy = self.raw_data["0" * self.num_qubits]

        popt, pcov = scipy.optimize.curve_fit(
            self.exp_decay,
            xx,
            yy,
            p0=(1.0 - 2**-self.num_qubits, 0.99, 2**-self.num_qubits),
            bounds=(0, 1),
            max_nfev=2000,
        )

        return popt, np.sqrt(np.diag(pcov))

    @staticmethod
    def exp_decay(
        x: np.typing.NDArray[np.float_], A: float, alpha: float, B: float
    ) -> np.typing.NDArray[np.float_]:
        r"""Exponential decay of the form

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

    ###################
    # Public Methods  #
    ###################
    def analyze_results(self, plot_results: bool = True) -> SSBResults:
        """Analyse the results and calculate the estimated CZ gate fidelity.

        Args:
            plot_results (optional): Whether to generate the data plots. Defaults to True.

        Returns:
           The final results from the experiment.
        """
        fit = self._fit_decay()
        cz_fidelity = fit[0][1]
        cz_fidelity_std = fit[1][1]

        if plot_results:
            self.plot_results()

        return SSBResults(
            target="& ".join(self.targets),
            total_circuits=len(self.samples),
            cz_fidelity_estimate=cz_fidelity,
            cz_fidelity_estimate_std=cz_fidelity_std,
        )

    def plot_results(self) -> None:
        """Plot the experiment data and the corresponding fits. The shaded upper and lower limits
        of the shaded region indicate the fits at +/- 1 standard deviation in all fitted parameters.
        """
        plot = sns.scatterplot(
            data=self.raw_data,
            x="num_cz_gates",
            y="0" * self.num_qubits,
        )
        fit = self._fit_decay()
        xx = np.linspace(0, np.max(self.raw_data.num_cz_gates))
        plot.plot(
            xx,
            self.exp_decay(xx, *fit[0]),
            color="tab:blue",
            linestyle="--",
        )
        plot.fill_between(
            xx,
            self.exp_decay(xx, *(fit[0] - fit[1])),
            self.exp_decay(xx, *(fit[0] + fit[1])),
            alpha=0.5,
            color="tab:blue",
        )
