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
"""Tooling for interleaved randomised benchmarking
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Union  # noqa: MDA400

import cirq
import cirq.circuits
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import seaborn as sns
from tqdm.auto import trange
from tqdm.contrib.itertools import product

from supermarq.qcvv.base_experiment import BenchmarkingExperiment, BenchmarkingResults, Sample


####################################################################################################
# Some handy functions for 1 and 2 qubit Clifford operations
####################################################################################################
def _reduce_single_qubit_clifford_seq(
    gate_seq: list[cirq.SingleQubitCliffordGate],
) -> cirq.SingleQubitCliffordGate:
    """Reduces a list of single qubit clifford gates to a single gate.

    Args:
        gate_seq: The list of gates.
    Returns:
        The single reduced gate.
    """
    cur = gate_seq[0]
    for gate in gate_seq[1:]:
        cur = cur.merged_with(gate)
    return cur


def _reduce_clifford_seq(
    gate_seq: list[cirq.CliffordGate],
) -> cirq.CliffordGate:
    """Reduces a list of multi qubit clifford gates to a single gate.

    Args:
        gate_seq: The list of gates.
    Returns:
        The single reduced gate.
    """
    cur = gate_seq[0].clifford_tableau
    for gate in gate_seq[1:]:
        cur = cur.then(gate.clifford_tableau)
    return cirq.CliffordGate.from_clifford_tableau(cur)


####################################################################################################
# The sets `S1`, `S1_X` and `S1_Y` of single qubit Clifford operations are used to generate
# random two qubit Clifford operations. For details see: https://arxiv.org/pdf/1210.7011 &
# https://arxiv.org/pdf/1402.4848.
# The implementation is adapted from:
# https://github.com/quantumlib/Cirq/blob/main/cirq-core/cirq/experiments/qubit_characterizations.py
####################################################################################################
_S1 = [
    # I
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, False], dtype=np.dtype("bool")),
            xs=np.array([[True], [False]], dtype=np.dtype("bool")),
            zs=np.array([[False], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # Y**0.5 X**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, False], dtype=np.dtype("bool")),
            xs=np.array([[True], [True]], dtype=np.dtype("bool")),
            zs=np.array([[True], [False]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # X**-0.5 Y**-0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, False], dtype=np.dtype("bool")),
            xs=np.array([[False], [True]], dtype=np.dtype("bool")),
            zs=np.array([[True], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
]


_S1_X = [
    # X**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, True], dtype=np.dtype("bool")),
            xs=np.array([[True], [True]], dtype=np.dtype("bool")),
            zs=np.array([[False], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # X**0.5, Y**0.5, X**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, True], dtype=np.dtype("bool")),
            xs=np.array([[True], [False]], dtype=np.dtype("bool")),
            zs=np.array([[True], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # Y**-0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([False, True], dtype=np.dtype("bool")),
            xs=np.array([[False], [True]], dtype=np.dtype("bool")),
            zs=np.array([[True], [False]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
]


_S1_Y = [
    # Y**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([True, False], dtype=np.dtype("bool")),
            xs=np.array([[False], [True]], dtype=np.dtype("bool")),
            zs=np.array([[True], [False]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # X**-0.5 Y**-0.5 X**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([True, False], dtype=np.dtype("bool")),
            xs=np.array([[True], [False]], dtype=np.dtype("bool")),
            zs=np.array([[True], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
    # Y X**0.5
    cirq.CliffordGate.from_clifford_tableau(
        cirq.CliffordTableau(
            1,
            rs=np.array([True, False], dtype=np.dtype("bool")),
            xs=np.array([[True], [True]], dtype=np.dtype("bool")),
            zs=np.array([[False], [True]], dtype=np.dtype("bool")),
            initial_state=0,
        )
    ),
]


@dataclass(frozen=True)
class IRBResults(BenchmarkingResults):
    """Data structure for the IRB experiment results."""

    rb_decay_coefficient: float
    """Decay coefficient estimate without the interleaving gate."""
    rb_decay_coefficient_std: float
    """Standard deviation of the decay coefficient estimate without the interleaving gate."""
    irb_decay_coefficient: float | None
    """Decay coefficient estimate with the interleaving gate."""
    irb_decay_coefficient_std: float | None
    """Standard deviation of the decay coefficient estimate with the interleaving gate."""
    average_interleaved_gate_error: float | None
    """Estimate of the interleaving gate error."""
    average_interleaved_gate_error_std: float | None
    """Standard deviation of the estimate for the interleaving gate error."""

    experiment_name = "IRB"


@dataclass(frozen=True)
class RBResults(BenchmarkingResults):
    """Data structure for the RB experiment results."""

    rb_decay_coefficient: float
    """Decay coefficient estimate without the interleaving gate."""
    rb_decay_coefficient_std: float
    """Standard deviation of the decay coefficient estimate without the interleaving gate."""
    average_error_per_clifford: float | None
    """Estimate of the average error per Clifford operation."""
    average_error_per_clifford_std: float | None
    """Standard deviation of the the average error per Clifford operation."""

    experiment_name = "RB"


class IRB(BenchmarkingExperiment[Union[IRBResults, RBResults]]):
    r"""Interleaved random benchmarking (IRB) experiment.

    IRB estimates the gate error of specified Clifford gate, :math:`\mathcal{C}^*`.
    This is achieved by first choosing a random sequence, :math:`\{\mathcal{C_i}\}_m`
    of :math:`m` Clifford gates and then using this to generate two circuits. The first
    is generated by appending to this sequence the single gate that corresponds to the
    inverse of the original sequence. The second circuit it obtained by inserting the
    interleaving gate, :math:`\mathcal{C}^*` after each gate in the sequence and then
    again appending the corresponding inverse element of the new circuit. Thus both
    circuits correspond to the identity operation.

    We run both circuits on the specified target and calculate the probability of measuring
    the resulting state in the ground state, :math:`p(0...0)`. This gives the circuit fidelity

    .. math::

        f(m) = 2p(0...0) - 1

    We can then fit an exponential decay :math:`\log(f) \sim m` to this circuit fidelity
    for each circuit, with decay rates :math:`\alpha` and :math:`\tilde{\alpha}` for the circuit
    without and with interleaving respectively. Finally the gate error of the
    specified gate, :math:`\mathcal{C}^*` is estimated as

    .. math::

        e_{\mathcal{C}^*} = \frac{1}{2} \left(1 - \frac{\tilde{\alpha}}{\alpha}\right)

    For more details see: https://arxiv.org/abs/1203.4550
    """

    def __init__(
        self,
        interleaved_gate: cirq.Gate | None = cirq.Z,
        num_qubits: int | None = 1,
        clifford_op_gateset: cirq.CompilationTargetGateset = cirq.CZTargetGateset(),
        *,
        random_seed: int | np.random.Generator | None = None,
    ) -> None:
        """Constructs an IRB experiment.

        Args:
            interleaved_gate: The Clifford gate to measure the gate error of. If None
                then no interleaving is performed and instead vanilla randomized benchmarking is
                performed.
            num_qubits: The number of qubits to experiment on. Must either be 1 or 2 but is ignored
                if a gate is provided - the number of qubits is instead inferred from the gate.
            clifford_op_gateset: The gateset to use when implementing the clifford operations.
                Defaults to the CZ/GR set.
            random_seed: An optional seed to use for randomization.
        """
        if interleaved_gate is not None:
            num_qubits = interleaved_gate.num_qubits()
        if num_qubits not in [1, 2]:
            raise NotImplementedError(
                "IRB experiment is currently only implemented for single or two qubit use."
            )
        super().__init__(num_qubits=num_qubits, random_seed=random_seed)

        if interleaved_gate is not None:
            self.interleaved_gate: cirq.CliffordGate | None = cirq.CliffordGate.from_op_list(
                [interleaved_gate(*self.qubits)], self.qubits
            )
            self._interleaved_op = interleaved_gate(*self.qubits)
            """The operation being interleaved"""
        else:
            self.interleaved_gate = None

        self.clifford_op_gateset = clifford_op_gateset
        """The gateset to use when implementing Clifford operations."""

    def _clifford_gate_to_circuit(
        self,
        clifford: cirq.CliffordGate,
    ) -> cirq.Circuit:
        """Converts a Clifford gate to a circuit using the desired gateset for the experiment.

        Args:
            clifford: The clifford operation to convert.

        Returns:
            A circuit implementing the desired Clifford gate.
        """
        circuit = cirq.Circuit(
            cirq.decompose_clifford_tableau_to_operations(
                self.qubits, clifford.clifford_tableau  # type: ignore[arg-type]
            )
        )
        return cirq.optimize_for_target_gateset(circuit, gateset=self.clifford_op_gateset)

    def random_single_qubit_clifford(self) -> cirq.SingleQubitCliffordGate:
        """Choose a random single qubit clifford gate.

        Returns:
            The random clifford gate.
        """
        Id = cirq.SingleQubitCliffordGate.I
        H = cirq.SingleQubitCliffordGate.H
        S = cirq.SingleQubitCliffordGate.Z_sqrt
        X = cirq.SingleQubitCliffordGate.X
        Y = cirq.SingleQubitCliffordGate.Y
        Z = cirq.SingleQubitCliffordGate.Z

        set_A = np.array(
            [
                Id,
                S,
                H,
                _reduce_single_qubit_clifford_seq([H, S]),
                _reduce_single_qubit_clifford_seq([S, H]),
                _reduce_single_qubit_clifford_seq([H, S, H]),
            ]
        )

        set_B = np.array([Id, X, Y, Z])

        return _reduce_single_qubit_clifford_seq([self._rng.choice(set_A), self._rng.choice(set_B)])

    def random_two_qubit_clifford(self) -> cirq.CliffordGate:
        """Choose a random two qubit clifford gate.

        For algorithm details see https://arxiv.org/pdf/1402.4848 & https://arxiv.org/pdf/1210.7011.

        Returns:
            The random clifford gate.
        """
        qubits = cirq.LineQubit.range(2)
        a = self.random_single_qubit_clifford()
        b = self.random_single_qubit_clifford()
        idx = self._rng.integers(20)
        if idx == 0:
            return cirq.CliffordGate.from_op_list([a(qubits[0]), b(qubits[1])], qubits)
        elif idx == 1:
            return cirq.CliffordGate.from_op_list(
                [
                    a(qubits[0]),
                    b(qubits[1]),
                    cirq.CZ(*qubits),
                    cirq.Y(qubits[0]) ** -0.5,
                    cirq.Y(qubits[1]) ** 0.5,
                    cirq.CZ(*qubits),
                    cirq.Y(qubits[0]) ** 0.5,
                    cirq.Y(qubits[1]) ** -0.5,
                    cirq.CZ(*qubits),
                    cirq.Y(qubits[1]) ** 0.5,
                ],
                qubits,
            )
        elif 2 <= idx <= 10:
            idx_a = int((idx - 2) / 3)
            idx_b = (idx - 2) % 3
            return cirq.CliffordGate.from_op_list(
                [
                    a(qubits[0]),
                    b(qubits[1]),
                    cirq.CZ(*qubits),
                    _S1[idx_a](qubits[0]),
                    _S1_Y[idx_b](qubits[1]),
                ],
                qubits,
            )

        idx_a = int((idx - 11) / 3)
        idx_b = (idx - 11) % 3
        return cirq.CliffordGate.from_op_list(
            [
                a(qubits[0]),
                b(qubits[1]),
                cirq.CZ(*qubits),
                cirq.Y(qubits[0]) ** 0.5,
                cirq.X(qubits[1]) ** -0.5,
                cirq.CZ(*qubits),
                _S1_Y[idx_a](qubits[0]),
                _S1_X[idx_b](qubits[1]),
            ],
            qubits,
        )

    def random_clifford(self) -> cirq.CliffordGate:
        """Returns:
        A random clifford gate with the correct number of qubits for the current experiment.
        """
        if self.num_qubits == 1:
            return self.random_single_qubit_clifford()
        return self.random_two_qubit_clifford()

    def gates_per_clifford(self, samples: int = 500) -> dict[str, float]:
        """Samples a number of random Clifford operations and calculates the average number of
        single and two qubit gates used to implement them. Note this depends on the gateset chosen
        for the experiment.

        Args:
            samples: Number of samples to use. Defaults to 500.

        Returns:
            A dictionary with the average number of one and two qubit gates used.
        """
        sample = [
            self._clifford_gate_to_circuit(self.random_clifford())
            for _ in trange(samples, desc="Sampling Clifford operations")
        ]
        return {
            "single_qubit_gates": np.mean(
                [
                    sum(1 for op in circuit.all_operations() if len(op.qubits) == 1)
                    for circuit in sample
                ]
            ).item(),
            "two_qubit_gates": np.mean(
                [
                    sum(1 for op in circuit.all_operations() if len(op.qubits) == 2)
                    for circuit in sample
                ]
            ).item(),
        }

    def _build_circuits(self, num_circuits: int, cycle_depths: Iterable[int]) -> Sequence[Sample]:
        """Build a list of randomised circuits required for the IRB experiment.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different cycle depths to use during the experiment.

        Returns:
            The list of experiment samples.
        """
        samples = []
        for _, depth in product(range(num_circuits), cycle_depths, desc="Building circuits"):
            base_sequence = [self.random_clifford() for _ in range(depth)]
            rb_sequence = base_sequence + [
                _reduce_clifford_seq(cirq.inverse(base_sequence))  # type: ignore[arg-type]
            ]
            rb_circuit = cirq.Circuit(self._clifford_gate_to_circuit(gate) for gate in rb_sequence)
            samples.append(
                Sample(
                    raw_circuit=rb_circuit + cirq.measure(sorted(self.qubits)),
                    data={
                        "num_cycles": depth,
                        "circuit_depth": len(rb_circuit),
                        "single_qubit_gates": sum(
                            1 for op in rb_circuit.all_operations() if len(op.qubits) == 1
                        ),
                        "two_qubit_gates": sum(
                            1 for op in rb_circuit.all_operations() if len(op.qubits) == 2
                        ),
                        "experiment": "RB",
                    },
                ),
            )
            if self.interleaved_gate is not None:
                # Find final gate
                irb_sequence = [elem for x in base_sequence for elem in (x, self.interleaved_gate)]
                irb_sequence_final_gate = _reduce_clifford_seq(
                    cirq.inverse(irb_sequence)  # type: ignore[arg-type]
                )

                irb_circuit = cirq.Circuit()
                for gate in base_sequence:
                    irb_circuit += self._clifford_gate_to_circuit(gate)
                    irb_circuit += self._interleaved_op.with_tags("no_compile")
                # Add the final inverting gate
                irb_circuit += self._clifford_gate_to_circuit(
                    irb_sequence_final_gate,
                )

                samples.append(
                    Sample(
                        raw_circuit=irb_circuit + cirq.measure(sorted(self.qubits)),
                        data={
                            "num_cycles": depth,
                            "circuit_depth": len(irb_circuit),
                            "single_qubit_gates": sum(
                                1 for op in irb_circuit.all_operations() if len(op.qubits) == 1
                            ),
                            "two_qubit_gates": sum(
                                1 for op in irb_circuit.all_operations() if len(op.qubits) == 2
                            ),
                            "experiment": "IRB",
                        },
                    ),
                )
        return samples

    def _process_probabilities(self, samples: Sequence[Sample]) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into the data structures
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """

        records = []
        missing_count = 0  # Count the number of samples that do not have probabilities saved
        for sample in samples:
            if sample.probabilities is not None:
                records.append(
                    {
                        "clifford_depth": sample.data["num_cycles"],
                        "circuit_depth": sample.data["circuit_depth"],
                        "experiment": sample.data["experiment"],
                        "single_qubit_gates": sample.data["single_qubit_gates"],
                        "two_qubit_gates": sample.data["two_qubit_gates"],
                        **sample.probabilities,
                    }
                )
            else:
                missing_count += 1

        if missing_count > 0:
            warnings.warn(
                f"{missing_count} sample(s) are missing probabilities. "
                "These samples have been omitted."
            )

        return pd.DataFrame(records)

    def plot_results(self) -> None:
        """Plot the exponential decay of the circuit fidelity with cycle depth."""
        plot = sns.scatterplot(
            data=self.raw_data,
            x="clifford_depth",
            y="0" * self.num_qubits,
            hue="experiment",
        )
        plot.set_xlabel(r"Cycle depth", fontsize=15)
        plot.set_ylabel(r"Survival probability", fontsize=15)
        plot.set_title(r"Exponential decay of survival probability", fontsize=15)

        rb_fit = self._fit_decay()
        xx = np.linspace(0, np.max(self.raw_data.clifford_depth))
        plot.plot(
            xx,
            self.exp_decay(xx, *rb_fit[0]),
            color="tab:blue",
            linestyle="--",
        )
        plot.fill_between(
            xx,
            self.exp_decay(xx, *(rb_fit[0] - rb_fit[1])),
            self.exp_decay(xx, *(rb_fit[0] + rb_fit[1])),
            alpha=0.5,
            color="tab:blue",
        )
        if self.interleaved_gate is not None:
            irb_fit = self._fit_decay("IRB")
            xx = np.linspace(0, np.max(self.raw_data.clifford_depth))
            plot.plot(
                xx,
                self.exp_decay(xx, *irb_fit[0]),
                color="tab:orange",
                linestyle="--",
            )
            plot.fill_between(
                xx,
                self.exp_decay(xx, *(irb_fit[0] - irb_fit[1])),
                self.exp_decay(xx, *(irb_fit[0] + irb_fit[1])),
                alpha=0.5,
                color="tab:orange",
            )

    def analyze_results(self, plot_results: bool = True) -> IRBResults | RBResults:
        """Analyse the experiment results and estimate the interleaved gate error.

        Args:
            plot_results: Whether to generate plots of the results. Defaults to False.

        Returns:
            A named tuple of the final results from the experiment.
        """

        rb_fit = self._fit_decay("RB")
        rb_decay_coefficient, rb_decay_coefficient_std = rb_fit[0][1], rb_fit[1][1]

        if self.interleaved_gate is None:
            self._results = RBResults(
                target="& ".join(self.targets),
                total_circuits=len(self.samples),
                rb_decay_coefficient=rb_decay_coefficient,
                rb_decay_coefficient_std=rb_decay_coefficient_std,
                average_error_per_clifford=(1 - 2**-self.num_qubits) * (1 - rb_decay_coefficient),
                average_error_per_clifford_std=(1 - 2**-self.num_qubits) * rb_decay_coefficient_std,
            )

            if plot_results:
                self.plot_results()

            return self.results

        else:
            irb_fit = self._fit_decay("IRB")
            irb_decay_coefficient, irb_decay_coefficient_std = irb_fit[0][1], irb_fit[1][1]
            interleaved_gate_error = (1 - irb_decay_coefficient / rb_decay_coefficient) * (
                1 - 2**-self.num_qubits
            )

            interleaved_gate_error_std = (
                (1 - 2**-self.num_qubits) / rb_decay_coefficient
            ) * np.sqrt(
                irb_decay_coefficient_std**2
                + irb_decay_coefficient**2 * rb_decay_coefficient_std**2
            )

            self._results = IRBResults(
                target="& ".join(self.targets),
                total_circuits=len(self.samples),
                rb_decay_coefficient=rb_decay_coefficient,
                rb_decay_coefficient_std=rb_decay_coefficient_std,
                irb_decay_coefficient=irb_decay_coefficient,
                irb_decay_coefficient_std=irb_decay_coefficient_std,
                average_interleaved_gate_error=interleaved_gate_error,
                average_interleaved_gate_error_std=interleaved_gate_error_std,
            )

            if plot_results:
                self.plot_results()

            return self.results

    def _fit_decay(
        self, experiment: str = "RB"
    ) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
        """Fits the exponential decay function to either the RB or IRB results.

        Args:
            experiment: Either `RB` or `IRB` referring to which data to filter by. Defaults to "RB".

        Returns:
            Tuple of the decay fit parameters and their standard deviations.
        """

        xx = self.raw_data.query(f"experiment == '{experiment}'")["clifford_depth"]
        yy = self.raw_data.query(f"experiment == '{experiment}'")["0" * self.num_qubits]

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
    def exp_decay(x: npt.ArrayLike, A: float, alpha: float, B: float) -> npt.ArrayLike:
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
        return A * (np.asarray(alpha) ** x) + B
