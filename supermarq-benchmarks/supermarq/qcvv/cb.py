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

from supermarq.qcvv import BenchmarkingExperiment, BenchmarkingResults, Sample

# Single-qubit basis rotations.
STRING_TO_ROTATION = {"I": cirq.I, "X": cirq.Y**0.5, "Y": cirq.X ** (-0.5), "Z": cirq.I}
STRING_TO_PAULI = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}


@dataclass(frozen=True)
class CBResults(BenchmarkingResults):
    """Results from an CB experiment."""

    channel_fidelities: pd.DataFrame
    process_fidelity: float
    dressed_process: bool
    experiment_name = "CB"


class CB(BenchmarkingExperiment[CBResults]):
    def __init__(
        self,
        process_circuit: cirq.Circuit,
        dressed_measurement: bool = True,
        num_channels: int | None = None,
        pauli_channels: list[str] | None = None,
    ) -> None:

        super().__init__(num_qubits=cirq.num_qubits(process_circuit))
        self.qubits = process_circuit.all_qubits()
        # Checks that the process is a Clifford circuit
        check, compiled_circuit = CB._is_Clifford(process_circuit)
        if not check:
            raise Exception("This cycle benchmarking is only valid for Clifford elements.")
        self._process_circuit = cirq.Circuit()
        # Prevent internal compiler optimizations
        for op in process_circuit.all_operations():
            self._process_circuit += op.with_tags("no_compile")

        self._matrix_root = CB._find_process_identity_min_depth(compiled_circuit)

        # Choose the random channels to use: 
        if pauli_channels is not None:
            # Makes sure that the channels are distinct.
            self.pauli_channels = [
                (
                    channel,
                    cirq.Moment(
                        STRING_TO_PAULI[pauli](qubit)
                        for (pauli, qubit) in zip(channel, self.sorted_qubits)
                    ),
                )
                for channel in set(pauli_channels)
            ]
        elif num_channels is not None:
            self.pauli_channels = self._generate_n_qubit_pauli_moments(num_channels)
        else:
            raise RuntimeError("Cannot have both `num_channels` and `pauli_channels` be None.")

        self._dressed_measurement = dressed_measurement

    ##############
    # Properties #
    ##############
    @property
    def sorted_qubits(self) -> list[cirq.Qubit]:
        return sorted(list(self.qubits))

    ###################
    # Private Methods #
    ###################
    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: list[int],
    ) -> Sequence[Sample]:
        """Build a list of random circuits to perform the CB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.
                For CB the provided depths are multiplied by the
                matrix root of the process circuit.

        Returns:
            The list of experiment samples.
        """
        if len(cycle_depths) != 2:
            raise ValueError("cycle benchmarking requires exactly two cycle depths")  # Only needs two cycle depths.

        samples = []
        for channel, depth, _ in tqdm.contrib.itertools.product(
            self.pauli_channels, cycle_depths, range(num_circuits), desc="Building circuits"
        ):
            circuit, c_of_p = self._generate_full_cb_circuit(
                channel=channel[0], depth=depth * self._matrix_root
            )
            samples.append(
                Sample(
                    raw_circuit=circuit,
                    data={
                        "pauli_channel": channel[0],
                        "cycle_depth": depth * self._matrix_root,
                        "c_of_p": c_of_p,
                        "circuit": "process",
                    },
                )
            )
            if not self._dressed_measurement:  # Compare with identity process
                circuit, c_of_p = self._generate_full_cb_circuit(
                    channel=channel[0], depth=depth * self._matrix_root, process=False
                )
                samples.append(
                    Sample(
                        raw_circuit=circuit,
                        data={
                            "pauli_channel": channel[0],
                            "cycle_depth": depth * self._matrix_root,
                            "c_of_p": c_of_p,
                            "circuit": "identity",
                        },
                    )
                )

        return samples

    def _cb_bulk_circuit(
        self, depth: int, process: bool = True
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """TODO"""
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
        for ii in range(0, depth):
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
    def _is_Clifford(circuit: cirq.Circuit) -> tuple[bool, cirq.Circuit]:
        """Checks if the circuit is a Clifford circuit by compiling it to CZ and
        rotations gates and checking that all operations are stabilizer operations.
        Args:
            circuit: The circuit to check.
        Returns:
            A tuple containing a `bool` indicating if the circuit is a Clifford circuit 
            and the `cirq.Cicuit` compiled circuit.
        """
        compiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
        check = True
        for op in compiled_circuit.all_operations():
            check &= cirq.has_stabilizer_effect(op)
        return check, compiled_circuit
    
    @staticmethod
    def _apply_circuit_to_tableau(circuit: cirq.Circuit, tableau: cirq.CliffordTableau) -> None:
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

    # @staticmethod
    # def _find_process_identity_min_depth(circuit, max_depth: int = 50) -> int:
    #     """TODO"""
    #     identity = np.identity(2 ** len(circuit.all_qubits()), dtype=complex)
    #     process_unitary = cirq.unitary(circuit)
    #     unitary = np.array(1 + 0j)
    #     for i in range(max_depth):
    #         unitary = np.dot(process_unitary, unitary)
    #         if cirq.equal_up_to_global_phase(unitary, identity):
    #             return i + 1
    #     raise RuntimeError(f"Could not find a circuit root less than {max_depth}")

    def _generate_full_cb_circuit(
        self, channel: str, depth: int, process=True
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Creates the full Cycle Benchmarking circuit.

        Args:
            channel: String representing the Pauli string of the particular
                Pauli eigenbasis channel.
            depth: Integer representing the number of repeated noisy implementations
                of the process circuit combined with random Pauli cycle layers.

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

    def _generate_n_qubit_pauli_moments(
        self,
        num_channels: int = 1,
    ) -> list[tuple[str, cirq.Moment]]:
        """Generates an n-qubit random Pauli sequence.

        Returns:
            Returns a `tuple` object of n-qubit pauli strings and their
            corresponding gate operations.
        """
        pauli_strings: list[str] = []
        pauli_moments: list[cirq.Moment] = []
        for _ in range(num_channels):
            paulis: list[str] = random.choices(list(STRING_TO_PAULI.keys()), k=self.num_qubits)
            pauli_string: str = "".join(paulis)
            while pauli_string in pauli_strings:
                paulis = random.choices(list(STRING_TO_PAULI.keys()), k=self.num_qubits)
                pauli_string = "".join(paulis)
            pauli_strings.append(pauli_string)
            pauli_moment: cirq.Moment = cirq.Moment(
                STRING_TO_PAULI[pauli](qubit) for (pauli, qubit) in zip(paulis, self.sorted_qubits)
            )
            pauli_moments.append(pauli_moment)
        return list(zip(pauli_strings, pauli_moments))
    
    def _inversion_circuit(
        self,
        channel_pauli_matrix: cirq.MutablePauliString[cirq.Qid],
        aggregate_pauli_superoperator: cirq.MutablePauliString[cirq.Qid],
    ) -> tuple[cirq.Circuit, cirq.MutablePauliString[cirq.Qid]]:
        """Creates the last layer of the Cycle Benchmarking circuit containing basis changing
        operations and the C(P) superoperator.

        Args:
            channel_pauli_matrix: `cirq.MutablePauliString` object representing the Pauli string of
            the particular Pauli eigenbasis channel.
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

    def _state_prep_circuit(self, channel: str) -> cirq.Circuit:
        """ """

        # This creates the state prep circuit to and rotates into
        # a particular Pauli eigenbasis channel.
        # Resulting state should be +1 eigenstate of pauli_matrix below.
        channel_basis_circuit = cirq.Circuit()
        channel_basis_circuit.append(
            [STRING_TO_ROTATION[s](self.sorted_qubits[ii]) for ii, s in enumerate(channel)]
        )
        channel_pauli_matrix: cirq.MutablePauliString[cirq.Qid] = cirq.MutablePauliString(
            cirq.Moment(
                STRING_TO_PAULI[pauli](qubit) for (pauli, qubit) in zip(channel, self.sorted_qubits)
            )
        )
        return channel_basis_circuit, channel_pauli_matrix

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
                    "pauli_channel": sample.data["pauli_channel"],
                    "cycle_depth": sample.data["cycle_depth"],
                    "expectation": self._sequence_expectation_value(
                        sample.data["c_of_p"],
                        sample.probabilities,
                    ),
                    "circuit": sample.data["circuit"],
                    **sample.probabilities,
                }
            )
        return pd.DataFrame(records)

    def _sequence_expectation_value(
        self, c_of_p: cirq.MutablePauliString[cirq.Qid], probabilities: dict[str, float]
    ) -> float:
        """Computes expectation values for each sequence.

        Args:
            c_of_p: Operator C(P).
            probabilities: TODO

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
                qubit_position_index = self.sorted_qubits.index(qubit)
                state_phase *= (-1) ** int(bit_string[qubit_position_index])
            expectation_value += state_phase * probs
        conj: complex = np.conj(global_phase)
        expectation_value *= conj
        return expectation_value.real

    ###################
    # Public Methods  #
    ###################
    def analyze_results(self, plot_results: bool = True) -> CBResults:
        """Analyse the results and calculate the estimated circuit fidelity.

        Args:
            plot_results (optional): Whether to generate the data plots. Defaults to True.

        Returns:
           The final results from the experiment.
        """
        m = self.raw_data.cycle_depth.unique()
        df = (
            self.raw_data.groupby(["pauli_channel", "circuit", "cycle_depth"])
            .sum()
            .reset_index()
            .pivot(columns=["cycle_depth", "circuit"], values="expectation", index="pauli_channel")
        )
        fidelities = (
            ((df[m[0]] / df[m[1]]) ** (1 / (m[0] - m[1]))).reset_index().rename_axis("", axis=1)
        )
        if self._dressed_measurement:
            fidelities.rename(columns={"process": "fidelity"}, inplace=True)
        else:
            fidelities["fidelity"] = fidelities["process"] / fidelities["identity"]
            fidelities.drop(columns=["process", "identity"], inplace=True)

        process_fidelity = fidelities["fidelity"].mean()

        if plot_results:
            self.plot_results()

        return CBResults(
            "$ ".join(self.targets),
            total_circuits=len(self.samples),
            channel_fidelities=fidelities,
            process_fidelity=process_fidelity,
            dressed_process=self._dressed_measurement,
        )

    def plot_results(self) -> None:
        """Plot the experiment data and the corresponding fits."""
