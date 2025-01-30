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
"""Tooling for interleaved randomised benchmarking"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cirq
import cirq.circuits
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import seaborn as sns
from tqdm.auto import trange
from tqdm.contrib.itertools import product

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample

if TYPE_CHECKING:
    from typing_extensions import Self


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
# random two qubit Clifford operations. For details see: https://arxiv.org/abs/1210.7011 &
# https://arxiv.org/abs/1402.4848.
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


@dataclass
class _RBResultsBase(QCVVResults):
    _rb_decay_coefficient: float | None = None
    """Decay coefficient estimate without the interleaving gate."""
    _rb_decay_coefficient_std: float | None = None
    """Standard deviation of the decay coefficient estimate without the interleaving gate."""

    @property
    def rb_decay_coefficient(self) -> float:
        """Returns:
        Decay coefficient estimate without the interleaving gate."""
        if self._rb_decay_coefficient is None:
            raise self._not_analyzed
        return self._rb_decay_coefficient

    @property
    def rb_decay_coefficient_std(self) -> float:
        """Returns:
        Standard deviation of the decay coefficient estimate without the interleaving gate."""
        if self._rb_decay_coefficient_std is None:
            raise self._not_analyzed
        return self._rb_decay_coefficient_std

    def _fit_decay(
        self, experiment: str = "RB"
    ) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
        """Fits the exponential decay function to either the RB or IRB results.

        Args:
            experiment: Either `RB` or `IRB` referring to which data to filter by. Defaults to "RB".

        Returns:
            Tuple of the decay fit parameters and their standard deviations.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot perform fit.")

        xx = self.data.query(f"experiment == '{experiment}'")["clifford_depth"]
        yy = self.data.query(f"experiment == '{experiment}'")["0" * self.num_qubits]

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

    def _plot_results(self) -> plt.Axes:
        """Plot the exponential decay of the circuit fidelity with cycle depth.

        Returns:
            A matplotlib axiss containing the RB decay plots and the corresponding
            fit.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot make plot.")
        plot = sns.scatterplot(
            data=self.data,
            x="clifford_depth",
            y="0" * self.num_qubits,
            hue="experiment",
        )
        plot.set_xlabel(r"Cycle depth", fontsize=15)
        plot.set_ylabel(r"Survival probability", fontsize=15)
        plot.set_title(r"Exponential decay of survival probability", fontsize=15)

        rb_fit = self._fit_decay()
        xx = np.linspace(0, np.max(self.data.clifford_depth))
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

        return plot

    def _analyze(self) -> None:
        rb_fit = self._fit_decay("RB")
        self._rb_decay_coefficient, self._rb_decay_coefficient_std = rb_fit[0][1], rb_fit[1][1]


@dataclass
class IRBResults(_RBResultsBase):
    """Data structure for the IRB experiment results."""

    _irb_decay_coefficient: float | None = None
    """Decay coefficient estimate with the interleaving gate."""
    _irb_decay_coefficient_std: float | None = None
    """Standard deviation of the decay coefficient estimate with the interleaving gate."""
    _average_interleaved_gate_error: float | None = None
    """Estimate of the interleaving gate error."""
    _average_interleaved_gate_error_std: float | None = None
    """Standard deviation of the estimate for the interleaving gate error."""

    @property
    def irb_decay_coefficient(self) -> float:
        """Returns:
        Decay coefficient estimate with the interleaving gate."""
        if self._irb_decay_coefficient is None:
            raise self._not_analyzed
        return self._irb_decay_coefficient

    @property
    def irb_decay_coefficient_std(self) -> float:
        """Returns:
        Standard deviation of the decay coefficient estimate with the interleaving gate."""
        if self._irb_decay_coefficient_std is None:
            raise self._not_analyzed
        return self._irb_decay_coefficient_std

    @property
    def average_interleaved_gate_error(self) -> float:
        """Returns:
        Estimate of the interleaving gate error."""
        if self._average_interleaved_gate_error is None:
            raise self._not_analyzed
        return self._average_interleaved_gate_error

    @property
    def average_interleaved_gate_error_std(self) -> float:
        """Returns:
        Standard deviation of the estimate for the interleaving gate error."""
        if self._average_interleaved_gate_error_std is None:
            raise self._not_analyzed
        return self._average_interleaved_gate_error_std

    def plot_results(
        self,
        filename: str | None = None,
    ) -> plt.Figure:
        """Plot the exponential decay of the circuit fidelity with cycle depth.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A single matplotlib figure containing the IRB and RB decay plots and the corresponding
            fits.

        Raises:
            RuntimeError: If no data is stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot make plot.")
        plot = self._plot_results()
        irb_fit = self._fit_decay("IRB")
        xx = np.linspace(0, np.max(self.data.clifford_depth))
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

        root_figure = plot.figure.figure
        if filename is not None:
            root_figure.savefig(filename, bbox_inches="tight")
        return root_figure

    def _analyze(self) -> None:
        super()._analyze()

        irb_fit = self._fit_decay("IRB")
        irb_decay_coefficient, irb_decay_coefficient_std = irb_fit[0][1], irb_fit[1][1]
        interleaved_gate_error = (1 - irb_decay_coefficient / self.rb_decay_coefficient) * (
            1 - 2**-self.num_qubits
        )

        interleaved_gate_error_std = (
            (1 - 2**-self.num_qubits) / self.rb_decay_coefficient
        ) * np.sqrt(
            irb_decay_coefficient_std**2
            + irb_decay_coefficient**2 * self.rb_decay_coefficient_std**2
        )

        self._irb_decay_coefficient = irb_decay_coefficient
        self._irb_decay_coefficient_std = irb_decay_coefficient_std
        self._average_interleaved_gate_error = interleaved_gate_error
        self._average_interleaved_gate_error_std = interleaved_gate_error_std

    def print_results(self) -> None:
        print(
            f"Estimated gate error: {self.average_interleaved_gate_error:.6f} +/- "
            f"{self.average_interleaved_gate_error_std:.6f}"
        )


@dataclass
class RBResults(_RBResultsBase):
    """Data structure for the RB experiment results."""

    _average_error_per_clifford: float | None = None
    """Estimate of the average error per Clifford operation."""
    _average_error_per_clifford_std: float | None = None
    """Standard deviation of the the average error per Clifford operation."""

    @property
    def average_error_per_clifford(self) -> float:
        """Returns:
        Estimate of the average error per Clifford operation."""
        if self._average_error_per_clifford is None:
            raise self._not_analyzed
        return self._average_error_per_clifford

    @property
    def average_error_per_clifford_std(self) -> float:
        """Returns:
        Standard deviation of the the average error per Clifford operation."""
        if self._average_error_per_clifford_std is None:
            raise self._not_analyzed
        return self._average_error_per_clifford_std

    def plot_results(
        self,
        filename: str | None = None,
    ) -> plt.Figure:
        """Plot the exponential decay of the circuit fidelity with cycle depth.

        Args:
            filename: Optional argument providing a filename to save the plots to. Defaults to None,
                indicating not to save the plot.

        Returns:
            A single matplotlib figure containing the RB decay plot and the corresponding fit.

        Raises:
            RuntimeError: If no data is stored.
        """
        if self.data is None:
            raise RuntimeError("No data stored. Cannot make plot.")

        plot = self._plot_results()
        root_figure = plot.figure.figure
        if filename is not None:
            root_figure.savefig(filename, bbox_inches="tight")
        return root_figure

    def _analyze(self) -> None:
        super()._analyze()
        self._average_error_per_clifford = (1 - 2**-self.num_qubits) * (
            1 - self.rb_decay_coefficient
        )
        self._average_error_per_clifford_std = (
            1 - 2**-self.num_qubits
        ) * self.rb_decay_coefficient_std

    def print_results(self) -> None:

        print(
            f"Estimated error per Clifford: {self.average_error_per_clifford:.6f} +/- "
            f"{self.average_error_per_clifford_std:.6f}"
        )


class IRB(QCVVExperiment[_RBResultsBase]):
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
        num_circuits: int,
        cycle_depths: Iterable[int],
        interleaved_gate: cirq.Gate | None = cirq.Z,
        num_qubits: int | None = 1,
        clifford_op_gateset: cirq.CompilationTargetGateset = cirq.CZTargetGateset(),
        *,
        random_seed: int | np.random.Generator | None = None,
        _samples: list[Sample] | None = None,
        **kwargs: str,
    ) -> None:
        """Constructs an IRB experiment.

        Args:
            num_circuits: Number of circuits to sample.
            cycle_depths: The cycle depths to sample.
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

        if interleaved_gate is not None:
            self.interleaved_gate: cirq.CliffordGate | None = cirq.CliffordGate.from_op_list(
                [interleaved_gate(*cirq.LineQubit.range(num_qubits))],
                cirq.LineQubit.range(num_qubits),
            )
        else:
            self.interleaved_gate = None

        self.clifford_op_gateset = clifford_op_gateset
        """The gateset to use when implementing Clifford operations."""

        if self.interleaved_gate is None:
            results_cls: type[RBResults] | type[IRBResults] = RBResults
        else:
            results_cls = IRBResults

        super().__init__(
            num_qubits=num_qubits,
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            random_seed=random_seed,
            results_cls=results_cls,
            _samples=_samples,
            **kwargs,
        )

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

        For algorithm details see https://arxiv.org/abs/1402.4848 & https://arxiv.org/abs/1210.7011.

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
        for k, depth in product(range(num_circuits), cycle_depths, desc="Building circuits"):
            base_sequence = [self.random_clifford() for _ in range(depth)]
            rb_sequence = base_sequence + [
                _reduce_clifford_seq(cirq.inverse(base_sequence))  # type: ignore[arg-type]
            ]
            rb_circuit = cirq.Circuit(self._clifford_gate_to_circuit(gate) for gate in rb_sequence)
            samples.append(
                Sample(
                    circuit=rb_circuit + cirq.measure(sorted(self.qubits)),
                    data={
                        "clifford_depth": depth,
                        "circuit_depth": len(rb_circuit),
                        "single_qubit_gates": sum(
                            1 for op in rb_circuit.all_operations() if len(op.qubits) == 1
                        ),
                        "two_qubit_gates": sum(
                            1 for op in rb_circuit.all_operations() if len(op.qubits) == 2
                        ),
                        "experiment": "RB",
                    },
                    circuit_realization=k,
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
                    irb_circuit += self.interleaved_gate(*self.qubits).with_tags("no_compile")
                # Add the final inverting gate
                irb_circuit += self._clifford_gate_to_circuit(
                    irb_sequence_final_gate,
                )

                samples.append(
                    Sample(
                        circuit=irb_circuit + cirq.measure(sorted(self.qubits)),
                        data={
                            "clifford_depth": depth,
                            "circuit_depth": len(irb_circuit),
                            "single_qubit_gates": sum(
                                1 for op in irb_circuit.all_operations() if len(op.qubits) == 1
                            ),
                            "two_qubit_gates": sum(
                                1 for op in irb_circuit.all_operations() if len(op.qubits) == 2
                            ),
                            "experiment": "IRB",
                        },
                        circuit_realization=k,
                    ),
                )
        return samples

    def _json_dict_(self) -> dict[str, Any]:
        """Converts the experiment to a json-able dictionary that can be used to recreate the
        experiment object. Note that the state of the random number generator is not stored.

        Returns:
            Json-able dictionary of the experiment data.
        """
        return {
            "interleaved_gate": self.interleaved_gate,
            "clifford_op_gateset": self.clifford_op_gateset,
            **super()._json_dict_(),
        }

    @classmethod
    def _from_json_dict_(
        cls,
        samples: list[Sample],
        interleaved_gate: cirq.Gate,
        clifford_op_gateset: cirq.CompilationTargetGateset,
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
        return cls(
            num_circuits=num_circuits,
            cycle_depths=cycle_depths,
            clifford_op_gateset=clifford_op_gateset,
            interleaved_gate=interleaved_gate,
            _samples=samples,
            **kwargs,
        )
