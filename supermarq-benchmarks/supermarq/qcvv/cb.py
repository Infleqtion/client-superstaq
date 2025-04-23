"""Tooling for cycle benchmarking experiments.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import cirq
import cirq.circuits
import numpy as np
import pandas as pd
import tqdm.contrib.itertools
import matplotlib.pyplot as plt

from supermarq.qcvv import QCVVExperiment, QCVVResults, Sample

# Single-qubit basis rotations.
STRING_TO_ROTATION = {"I": cirq.I, "X": cirq.Y**0.5, "Y": cirq.X**(-0.5), "Z": cirq.I}
STRING_TO_PAULI = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}


@dataclass
class CBResults(QCVVResults):
    """Results from a CB experiment."""
    """Fidelities of the Pauli channels."""
    _channel_fidelities: pd.DataFrame | None = None
    """Process fidelity estimate."""
    _process_fidelity_estimate: float | None = None
    """Standard deviation of the process fidelity estimate."""
    _process_fidelity_estimate_std: float | None = None
    """Undressed process fidelity estimate."""
    _undressed_process_fidelity_estimate: float | None = None
    """Standard deviation of the undressed process fidelity estimate."""
    _undressed_process_fidelity_estimate_std: float | None = None
    

    @property
    def channel_fidelities(self) -> pd.DataFrame:
        """Returns the channel fidelities."""
        if self._channel_fidelities is None:
            raise self._not_analyzed
        return self._channel_fidelities
    
    @property
    def process_fidelity(self) -> float:
        """Returns the process fidelity estimate."""
        if self._process_fidelity_estimate is None:
            raise self._not_analyzed
        return self._process_fidelity_estimate
    
    @property
    def process_fidelity_std(self) -> float:    
        """Returns the standard deviation of the process fidelity estimate."""
        if self._process_fidelity_estimate_std is None:
            raise self._not_analyzed
        return self._process_fidelity_estimate_std
    
    @property
    def undressed_process_fidelity(self) -> float:
        """Returns the undressed process fidelity estimate."""
        if self.experiment._undressed_process:
            if self._undressed_process_fidelity_estimate is None:
                raise self._not_analyzed
            return self._undressed_process_fidelity_estimate
        else:
            raise RuntimeError("Undressed process fidelity is not available for this experiment.")
    
    @property
    def process_fidelity_std(self) -> float:    
        """Returns the standard deviation of the undressed process fidelity estimate."""
        if self.experiment._undressed_process:
            if self._undressed_process_fidelity_estimate_std is None:
                raise self._not_analyzed
            return self._undressed_process_fidelity_estimate_std
        else:
            raise RuntimeError("Undressed process fidelity is not available for this experiment.")

    def plot_results(
        self,
        filename: str | None = None,
    ) -> plt.Figure:
        """Plot the experiment data and the corresponding fits."""
        expectations = self._channel_expectations
        fidelities = self._channel_fidelities
        processes = expectations.process.unique()
        fig, axs = plt.subplots(len(processes), 1, figsize=(10, 10))

        if not self.experiment._undressed_process:
            self._plot_decay("dressed", axs[0])
            self._plot_fidelities("dressed", axs[1])
        else:
            self._plot_decay("dressed", axs[0, 0])
            self._plot_fidelities("dressed", axs[0, 1])
            self._plot_decay("undressed", axs[1, 0])
            self._plot_fidelities("undressed", axs[1, 1])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        
        return fig
    
    def print_results(self) -> None:
        print(
            "Cycle Benchmarking Results:\n"
            f"Estimated process fidelity: {self._process_fidelity_estimate:.5}" 
            f" +/- {self._process_fidelity_estimate_std:.5}"
        )
        if self.experiment._undressed_process:
            print(
                f"Estimated undressed process fidelity: {self._undressed_process_fidelity_estimate:.5}"
                f" +/- {self._undressed_process_fidelity_estimate_std:.5}"
            )
    
    def _analyze(self) -> None:
        """Analyzes the results of the experiment and estimates the pauli channels
        and the process fidelities."""
        records = self._process_probabilities().rename(columns={"circuit": "process"})
        records.replace(["identity", "process"], ["undressed", "dressed"], inplace=True)
        depths = records.cycle_depth.unique()
        
        expectations = (
            records.groupby(["pauli_channel", "process", "cycle_depth"])
            .agg(
                expectation_mean=("expectation", "mean"),
                expectation_delta=("expectation", "std"),
            )
            .reset_index()
            .pivot(
                columns=["cycle_depth", "process"], 
                values=["expectation_mean", "expectation_delta"], 
                index="pauli_channel"
            )
        )

        fidelities = (
            ((expectations.expectation_mean[depths[0]] / 
            expectations.expectation_mean[depths[1]]) ** (1 / (depths[0] - depths[1])))
            .reset_index()
            .pivot(index="pauli_channel", columns=[])
        )

        fidelities_delta = (
            np.sqrt(
                (expectations.expectation_delta/expectations.expectation_mean)[depths[0]]**2+
                (expectations.expectation_delta/expectations.expectation_mean)[depths[1]]**2
            )
            .reset_index()
            .pivot(index="pauli_channel", columns=[])
        )

        fidelities_delta *= fidelities/abs(depths[0]- depths[1])
        fidelities.columns = pd.MultiIndex.from_product([["fidelity"], fidelities.columns])
        fidelities_delta.columns = pd.MultiIndex.from_product([["delta"], fidelities_delta.columns])

        if self.experiment._undressed_process:
            fidelities_delta.loc[:, ("delta", "undressed")] = np.sqrt(
                ((fidelities_delta.delta/fidelities.fidelity)**2)
                .sum(axis=1).values
            )
            fidelities.loc[:, ("fidelity", "undressed")] = (
                fidelities.fidelity.dressed/fidelities.fidelity.undressed
            )
            fidelities_delta.loc[:, ("delta", "undressed")] *= (
                fidelities.fidelity.undressed
            )
        
            for d in depths:
                expectations.loc[:, ("expectation_mean", d, "undressed")] = (
                    expectations.expectation_mean[d, "dressed"] / 
                    expectations.expectation_mean[d, "undressed"]
                )
                expectations.loc[:, ("expectation_delta", d, "undressed")] = (
                    np.sqrt(
                        (expectations.expectation_delta[d, "dressed"] /
                        expectations.expectation_mean[d, "dressed"])**2 +
                        (expectations.expectation_delta[d, "undressed"] /
                        expectations.expectation_mean[d, "undressed"])**2
                    )
                )

        fidelities = pd.concat([fidelities, fidelities_delta], axis=1)
        fidelities = fidelities.stack(level=[1], future_stack=True).reset_index()
        expectations = expectations.stack(level=[1,2], future_stack=True).reset_index()

        self._channel_fidelities = fidelities

        self._channel_expectations = expectations

        self._dressed_process_fidelity_estimate = (
                fidelities[fidelities.process=="dressed"].fidelity.mean()
            )
        self._dressed_process_fidelity_estimate_std = (
            np.sqrt(
                (fidelities[fidelities.process=="dressed"].delta**2).sum()
            )/len(fidelities)
        )

        if self.experiment._undressed_process:
            self._undressed_process_fidelity_estimate = (
                fidelities[fidelities.process=="undressed"].fidelity.mean()
            )
            self._undressed_process_fidelity_estimate_std = (
                np.sqrt(
                    (fidelities[fidelities.process=="undressed"].delta**2).sum()
                )/len(fidelities)
            )

    def _plot_decay(self, process: str, ax: plt.Axes) -> None:
        expectations = self._channel_expectations[
            self._channel_expectations.process==process
        ]
        fidelities = self._channel_fidelities[
            self._channel_fidelities.process==process
        ]
        pauli_channels = expectations.pauli_channel.unique()
        depths = expectations.cycle_depth.unique()
        for p_c in pauli_channels:
            ys = expectations[expectations.pauli_channel==p_c].expectation_mean
            ax.errorbar(
                x=depths, 
                y=ys,
                yerr=expectations[expectations.pauli_channel==p_c].expectation_delta,
                label=p_c,
                fmt="o",
                capsize=5
            )
            color = ax.get_lines()[-1].get_color()
            xs = np.linspace(0, 2*max(depths))
            a = (
                    expectations[
                        (expectations.pauli_channel==p_c) &
                        (expectations.cycle_depth==depths[0])
                    ].expectation_mean.iloc[0]/
                    (
                        fidelities[
                            (fidelities.pauli_channel==p_c)
                        ].fidelity.iloc[0]
                    )**depths[0]
                )
            reg = (
                a*fidelities[
                    (fidelities.pauli_channel==p_c)
                    ].fidelity.iloc[0]**xs
                )
            ax.plot(xs, reg, color=color)
        ax.set_xlabel("Number of cycles")
        ax.set_ylabel("Expectation value")
        ax.legend()

    def _plot_fidelities(self, process: str, ax: plt.Axes) -> None:
        fidelities = self._channel_fidelities[
            self._channel_fidelities.process == process
        ]
        if process == "dressed":
            estimate = self._dressed_process_fidelity_estimate
            std = self._dressed_process_fidelity_estimate_std
        else:
            estimate = self._undressed_process_fidelity_estimate
            std = self._undressed_process_fidelity_estimate_std
        ax.errorbar(
            x=fidelities.pauli_channel,
            y=fidelities.fidelity,
            yerr=fidelities.delta,
            fmt="D",
            capsize=5,
            label="Pauli fidelity"
        )
        ax.axhline(y=estimate, label="process fidelity")
        xs = np.linspace(*ax[1].get_xlim())
        ax.fill_between(
            xs, 
            estimate - std,
            estimate + std,
            alpha=0.2,
        )
        ax.set_xlabel("Pauli channel")
        ax.set_ylabel("Fidelity")
        ax.legend()

    def _process_probabilities(self) -> pd.DataFrame:
        """Processes the probabilities generated by sampling the circuits into the data structures
        needed for analyzing the results.

        Args:
            samples: The list of samples to process the results from.

        Returns:
            A data frame of the full results needed to analyse the experiment.
        """
        records = []
        for _, entry in self.data.iterrows():
            probabilities = dict(entry.filter(regex=f"[01]{{{self.num_qubits}}}"))
            records.append(
                {
                    "pauli_channel": entry.pauli_channel,
                    "cycle_depth": entry.cycle_depth,
                    "expectation": self._sequence_expectation_value(
                        entry.c_of_p,
                        probabilities,
                    ),
                    "circuit": entry.circuit,
                    **probabilities,
                }
            )
        return pd.DataFrame(records)

    def _sequence_expectation_value(
        self, 
        c_of_p: cirq.MutablePauliString[cirq.Qid], 
        probabilities: dict[str, float]
    ) -> float:
        """Computes expectation values for each sequence.

        Args:
            c_of_p: Operator C(P).
            probabilities: The probability distribution of the measurements.

        Returns:
            Returns the real value of the expectation value.
        """
        global_phase = complex(c_of_p.coefficient)
        expectation_value = complex(0.0)
        for bit_string, probs in probabilities.items():
            state_phase = 1
            for qubit in c_of_p:
                # Note that by virtue of a qubit being in c_of_p
                # it has a non-identity op. associated.
                qubit_position_index = self.experiment.sorted_qubits.index(qubit)
                state_phase *= (-1) ** int(bit_string[qubit_position_index])
            expectation_value += state_phase * probs
        conj: complex = np.conj(global_phase)
        expectation_value *= conj
        return expectation_value.real


class CB(QCVVExperiment[CBResults]):
    def __init__(
        self,
        process_circuit: cirq.Circuit,
        pauli_channels: int | list[str],
        num_circuits: int = 1,
        process_order_factor: list[int] = [1, 2],
        undressed_process: bool = False,
        *,
        random_seed: int | np.random.Generator | None = None,
        _samples: Sequence[Sample] | None = None,
        **kwargs,
    ) -> None:
        """Cycle Benchmarking experiment.
        Args:
            process_circuit: The circuit of the process to be benchmarked.
            pauli_channels: the number of Pauli channels to use or 
                a list of the Pauli channels to use.
            num_circuits: The number of circuits to generate per depth.
            process_order_factor: The factors by which to multiply the process order.
            undressed_process: Whether to estimate the undressed process fidelity.
            random_seed: An optional seed to use for randomization.
        """
        if len(process_order_factor) != 2:
            raise ValueError("cycle benchmarking only requires two factors")
   
        check, compiled_circuit = CB._is_Clifford(process_circuit)
        if not check:
            raise RuntimeError("This cycle benchmarking is only valid for Clifford elements.")
        
        self._matrix_order = self._find_process_order(compiled_circuit)

        cycle_depth = [f * self._matrix_order for f in process_order_factor]
        
        self._process_circuit = cirq.Circuit()
        # Prevent internal compiler optimizations
        for op in process_circuit.all_operations():
            self._process_circuit += op.with_tags("no_compile")

        self.qubits = list(process_circuit.all_qubits())
        self._num_qubits = len(self.qubits)

        if isinstance(pauli_channels, list):
            self.pauli_channels = []
            for channel in set(pauli_channels):
                if len(channel) != self._num_qubits:
                    raise RuntimeError(f"All Pauli channels must be over {self._num_qubits}" 
                                       f" qubits. {channel} is over {len(channel)} qubits.")
                self.pauli_channels.append(
                    (
                        channel,
                        cirq.Moment(
                            STRING_TO_PAULI[pauli](qubit)
                            for (pauli, qubit) in zip(channel, self.sorted_qubits)
                        ),
                    )
                )
        elif isinstance(pauli_channels, int):
            self.pauli_channels = self._generate_n_qubit_pauli_moments(pauli_channels)
        else:
            raise RuntimeError("`pauli_channels` must be a list of Pauli strings or an integer.")

        self._undressed_process = undressed_process

        super().__init__(
            qubits = self.qubits,
            num_circuits = num_circuits,
            cycle_depths = cycle_depth,
            random_seed = random_seed,
            results_cls = CBResults,
            _samples = _samples,
            **kwargs,
        )


    ##############
    # Properties #
    ##############
    @property
    def sorted_qubits(self) -> list[cirq.Qubit]:
        return sorted(self.qubits)

    ###################
    # Private Methods #
    ###################
    @staticmethod
    def _apply_circuit_to_tableau(
        circuit: cirq.Circuit, tableau: cirq.CliffordTableau
    ) -> None:
        """Turns the circuit into a Clifford tableau.

        Args:
            circuit: The circuit to express as a tableau."""
        sorted_qubits = sorted(circuit.all_qubits())
        for op in circuit.all_operations():
            if isinstance(op.gate, cirq.PhasedXZGate):
                tableau.apply_z(
                    axis=sorted_qubits.index(op.qubits[0]), 
                    exponent=-round(op.gate.axis_phase_exponent, 4)
                )
                tableau.apply_x(
                    axis=sorted_qubits.index(op.qubits[0]), 
                    exponent=round(op.gate.x_exponent, 4)
                )
                tableau.apply_z(
                    axis=sorted_qubits.index(op.qubits[0]), 
                    exponent=round(op.gate.axis_phase_exponent, 4)
                )
                tableau.apply_z(
                    axis=sorted_qubits.index(op.qubits[0]), 
                    exponent=round(op.gate.z_exponent, 4)
                )
            else:
                tableau.apply_cz(
                    control_axis=sorted_qubits.index(op.qubits[0]), 
                    target_axis=sorted_qubits.index(op.qubits[1]),
                    exponent=op.gate.exponent
                )

    def _build_circuits(
        self, num_circuits: int, cycle_depths: list[int]
    ) -> Sequence[Sample]:
        """Build a list of random circuits to perform the CB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.

        Returns:
            The list of experiment samples.
        """
        samples = []
        for channel, depth, k in tqdm.contrib.itertools.product(
            self.pauli_channels, cycle_depths, range(num_circuits), desc="Building circuits"
        ):
            circuit, c_of_p = self._generate_full_cb_circuit(
                channel=channel[0], depth=depth
            )
            samples.append(
                Sample(
                    circuit_realization=k,
                    circuit=circuit,
                    data={
                        "pauli_channel": channel[0],
                        "cycle_depth": depth,
                        "c_of_p": c_of_p,
                        "circuit": "process",
                    },
                )
            )
            if self._undressed_process:  # Compare with identity process
                circuit, c_of_p = self._generate_full_cb_circuit(
                    channel=channel[0], depth=depth, process=False
                )
                samples.append(
                    Sample(
                        circuit_realization=k,
                        circuit=circuit,
                        data={
                            "pauli_channel": channel[0],
                            "cycle_depth": depth,
                            "c_of_p": c_of_p,
                            "circuit": "identity",
                        },
                    )
                )

        return samples

    def _cb_bulk_circuit(
        self, depth: int, process: bool = True
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Creates the bulk circuit for CB by interleaving the noisy process with
        random Pauli elemts.

        Args:
            depth: The number of repeated noisy implementations
                of the process circuit interleaved with randomn Pauli cycle layers.
            process: Indicates whether to include the process circuit in the bulk circuit.

        Returns: a `tuple` containing the bulk `cirq.Circuit` and a
            `cirq.MutablePauliString` representing C(P).
        """
        # Create new bulk circuit
        bulk_circuit = cirq.Circuit()

        # Generate initial Pauli layer
        zeroth_moment = self._generate_n_qubit_pauli_moments()[0][1]

        # Append to bulk circuit and mutable Pauli string
        bulk_circuit.append(zeroth_moment)
        aggregate_pauli_string: cirq.MutablePauliString[cirq.Qid] = cirq.MutablePauliString(
            zeroth_moment
        )

        # Loop over interleaving Pauli layers
        for ii in range(depth):
            if process:
                bulk_circuit += self._process_circuit

            moment = self._generate_n_qubit_pauli_moments()[0][1]
            bulk_circuit.append(moment, strategy=cirq.circuits.InsertStrategy.NEW_THEN_INLINE)

            # Pauli string operations
            ith_pauli_string: cirq.MutablePauliString[cirq.Qid] = cirq.MutablePauliString(moment)
            if process:
                ith_pauli_string.inplace_before((ii + 1) * self._process_circuit)
            aggregate_pauli_string.inplace_right_multiply_by(ith_pauli_string)

        return bulk_circuit, aggregate_pauli_string
    
    @staticmethod
    def _find_process_order(circuit: cirq.Circuit, max_depth: int = 50) -> int:
        """Finds the order of the process via the Clifford tableau representation.

        Args:
            circuit: The circuit to find the order of.
            max_depth: The maximum depth to search for the order.

        Returns:
            The order of the process.
        """
        num_qubits = len(circuit.all_qubits())
        tableau = cirq.CliffordTableau(num_qubits)
        identity = np.eye(2*num_qubits, dtype=int)
        zeros = np.zeros(2*num_qubits, dtype=int)
        for i in range(max_depth):
            CB._apply_circuit_to_tableau(circuit, tableau)
            mat = tableau.matrix().astype(int)
            phases = tableau.rs.astype(int)
            if np.array_equal(mat, identity) and np.array_equal(phases, zeros):
                return i+1
        raise RuntimeError(f"Could not find a circuit order less than {max_depth}")
    
    def _from_json_dict_(self, json_dict: dict) -> CB:
        """Creates a Cycle Benchmarking experiment from a JSON dictionary."""
        process_circuit = cirq.read_json(json_dict["process_circuit"])
        pauli_channels = json_dict["pauli_channels"]
        num_circuits = json_dict["num_circuits"]
        process_order_factor = json_dict["process_order_factor"]
        undressed_process = json_dict["undressed_process"]
        random_seed = json_dict["random_seed"]
        return CB(
            process_circuit=process_circuit,
            pauli_channels=pauli_channels,
            num_circuits=num_circuits,
            process_order_factor=process_order_factor,
            undressed_process=undressed_process,
            random_seed=random_seed,
        )
    
    def _json_dict_(self) -> dict:
        """Creates a JSON dictionary from a Cycle Benchmarking experiment."""
        factors = [depth // self._matrix_order for depth in self.cycle_depths]
        return {
            "process_circuit": cirq.to_json(self._process_circuit),
            "pauli_channels": [channel[0] for channel in self.pauli_channels],
            "num_circuits": self.num_circuits,
            "process_order_factor": factors,
            "undressed_mprocess": self._undressed_process,
            "random_seed": self.random_seed,
        }

    def _generate_full_cb_circuit(
        self, channel: str, depth: int, process=True
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Creates the full Cycle Benchmarking circuit.

        Args:
            channel: String representing the Pauli string of the particular
                Pauli eigenbasis channel.
            depth: Integer representing the number of repeated noisy implementations
                of the process circuit interleaved with randomn Pauli cycle layers.

        Returns:
            Returns a `tuple` containing the CB `cirq.Circuit` and a
            `cirq.MutablePauliString` object representing C(P) superoperator.
        """
        state_prep_circuit, pauli_matrix = self._state_prep_circuit(channel)

        bulk_circuit, c_superoperator = self._cb_bulk_circuit(depth, process)

        inverse_circuit, c_of_p = self._inversion_circuit(pauli_matrix, c_superoperator)

        full_cb_circuit = cirq.Circuit(
            state_prep_circuit + bulk_circuit + inverse_circuit + cirq.measure(self.qubits)
        )
        return full_cb_circuit, c_of_p
    
    def _generate_all_pauli_strings(self) -> list[tuple[str, cirq.Moment]]:
        """Generates all possible Pauli strings of a given length.

        Returns:
            A list of tuples containing the Pauli string and the corresponding gate operations.
        """
        paulis = list(STRING_TO_PAULI.keys())
        pauli_strings: list[str] = []
        pauli_moments: list[cirq.Moment] = []
        for i in range(4**self._num_qubits):
            pauli_string = ""
            for j in range(self._num_qubits):
                pauli_string += paulis[i % 4]
                i //= 4
            pauli_strings.append(pauli_string)
            pauli_moment: cirq.Moment = cirq.Moment(
                STRING_TO_PAULI[pauli](qubit) for (pauli, qubit) in zip(paulis, self.sorted_qubits)
            )
            pauli_moments.append(pauli_moment)
        return list(zip(pauli_strings, pauli_moments))
    
    def _generate_random_pauli_strings(
        self, num_channels: int = 1
    ) -> list[tuple[str, cirq.Moment]]:
        """Generates distinct random Pauli strings of a given length.

        Args:
            num_channels: The number of Pauli strings to generate.

        Returns:
            A list of tuples containing the Pauli string and the corresponding 
            gate operations.
        """
        pauli_strings: list[str] = []
        pauli_moments: list[cirq.Moment] = []
        for _ in range(num_channels):
            paulis: list[str] = random.choices(list(STRING_TO_PAULI.keys()), k=self._num_qubits)
            pauli_string: str = "".join(paulis)
            while pauli_string in pauli_strings:
                paulis = random.choices(list(STRING_TO_PAULI.keys()), k=self._num_qubits)
                pauli_string = "".join(paulis)
            pauli_strings.append(pauli_string)
            pauli_moment: cirq.Moment = cirq.Moment(
                STRING_TO_PAULI[pauli](qubit) for (pauli, qubit) in zip(paulis, self.sorted_qubits)
            )
            pauli_moments.append(pauli_moment)
        return list(zip(pauli_strings, pauli_moments))

    def _generate_n_qubit_pauli_moments(
        self, num_channels: int = 1,
    ) -> list[tuple[str, cirq.Moment]]:
        """Generates distinct n-qubit random Pauli strings. If the number of
        channels is greater than the number of possible Pauli strings, it will
        generate all possible Pauli strings.

        Args:
            num_channels: Number of Pauli strings to generate.

        Returns:
            Returns a `list` of `tuple` object of n-qubit pauli strings and their
            corresponding gate operations.
        """
        if self._num_qubits*np.log(4) <= np.log(num_channels):
            # Generate all possible Pauli strings
            return self._generate_all_pauli_strings()
        else:
            # Generate distinct random Pauli strings
            return self._generate_random_pauli_strings(num_channels)
        
    def _inversion_circuit(
        self,
        channel_pauli_matrix: cirq.MutablePauliString[cirq.Qid],
        aggregate_pauli_superoperator: cirq.MutablePauliString[cirq.Qid],
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Creates the last layer of the Cycle Benchmarking circuit containing 
        basis changing operations and the C(P) superoperator.

        Args:
            channel_pauli_matrix: `cirq.MutablePauliString` object representing 
            the Pauli string of the particular Pauli eigenbasis channel.
            aggregate_pauli_superoperator: C(P) superoperator as a `cirq.MutablePauliString` object.

        Returns:
            Returns a `tuple` containing the inversion `cirq.Circuit` final layer and a
            `cirq.MutablePauliString` object representing C(P) superoperator.
        """
        pauli_to_inv_rot = {cirq.X: cirq.Y ** (-0.5), cirq.Y: cirq.X**0.5, cirq.Z: cirq.I}
        # Following few lines should construct operator C(P) = C P C^{\dag}
        channel_pauli_matrix.inplace_right_multiply_by(
            aggregate_pauli_superoperator
        )  # "Circuit right" = C P
        aggregate_pauli_phase = complex(aggregate_pauli_superoperator.coefficient)
        if aggregate_pauli_phase.real == 0:
            aggregate_pauli_superoperator *= (
                -1
            )  # Has the effect of conjugating coefficient if it's imaginary
        channel_pauli_matrix.inplace_left_multiply_by(
            aggregate_pauli_superoperator
        )  # "Circuit left" => (C P) C^{\dag}

        inverse_circuit = cirq.Circuit()
        inverse_circuit.append([pauli_to_inv_rot[v](k) for k, v in channel_pauli_matrix.items()])

        return inverse_circuit, channel_pauli_matrix  # Returns B^{\dagger} layer and C(P)
    
    @staticmethod
    def _is_Clifford(circuit: cirq.Circuit) -> tuple[bool, cirq.Circuit]:
        """Checks if the circuit is a Clifford circuit by compiling it to CZ and
        rotations gates and checking that all operations are stabilizer operations.

        Args:
            circuit: The circuit to check.

        Returns:
            A tuple containing a `bool` indicating if the circuit is a Clifford circuit 
            and the `cirq.Cicuit` compiled circuit.
        """
        compiled_circuit = cirq.optimize_for_target_gateset(
            circuit, 
            gateset=cirq.CZTargetGateset()
        )
        check = True
        for op in compiled_circuit.all_operations():
            check &= cirq.has_stabilizer_effect(op)
        return check, compiled_circuit
    
    def _state_prep_circuit(
        self, channel: str
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Prepares the initial state for CB, which is the +1 eigenstate of the
        Pauli channel.

        Args:
            channel: The Pauli channel of which the +1 eigenstate is prepared.

        Returns:
            Returns a `tuple` object containing the `cirq.Circuit` resulting 
            in the eigenstate and a `cirq.MutablePauliString` object 
            representing the Pauli string.
        """
        channel_basis_circuit = cirq.Circuit()
        channel_basis_circuit.append(
            [STRING_TO_ROTATION[s](self.sorted_qubits[ii]) for ii, s in enumerate(channel)]
        )
        channel_pauli_matrix: cirq.MutablePauliString[cirq.Qid] = cirq.MutablePauliString(
            cirq.Moment(
                STRING_TO_PAULI[s](self.sorted_qubits[ii]) for ii, s in enumerate(channel)
            )
        )
        return channel_basis_circuit, channel_pauli_matrix

    
