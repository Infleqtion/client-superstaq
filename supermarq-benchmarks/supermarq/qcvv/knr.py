"""Tooling for K-body noise reconstruction experiments.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import cirq
import numpy as np
import pandas as pd
import tqdm.contrib.itertools

from supermarq.qcvv import BenchmarkingExperiment, BenchmarkingResults, Sample

# Single-qubit basis rotations.
_STRING_TO_ROTATION = {"I": cirq.I, "X": cirq.Y**0.5, "Y": cirq.X ** (-0.5), "Z": cirq.I}
_STRING_TO_PAULI = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
_PAULI_INDEX = ["I", "X", "Y", "Z"]


@dataclass(frozen=True)
class KNRResults(BenchmarkingResults):
    """Results from an CB experiment."""

    channel_fidelities: pd.DataFrame
    process_fidelity: float
    dressed_process: bool
    experiment_name = "KNR"


class KNR(BenchmarkingExperiment[KNRResults]):
    """Follows protocol 3 of https://arxiv.org/pdf/2303.17714v1."""

    def __init__(
        self,
        process_circuit: cirq.Circuit,
        num_parallel_supports: int = 2,
    ) -> None:
        super().__init__(num_qubits=cirq.num_qubits(process_circuit))
        self.qubits = process_circuit.all_qubits()

        # TODO: Add check for Clifford process
        self._process_circuit = cirq.Circuit()
        # Prevent internal compiler optimizations
        for op in process_circuit.all_operations():
            self._process_circuit += op.with_tags("no_compile")

        self._matrix_root = KNR._find_process_identity_min_depth(process_circuit)

        self._support_channels = self._channels_to_measure(num_parallel_supports)

        self.pauli_channels = list(itertools.chain.from_iterable(self._support_channels.values()))

    ##############
    # Properties #
    ##############
    @property
    def sorted_qubits(self) -> list[cirq.Qubit]:
        return sorted(list(self.qubits))

    ###################
    # Private Methods #
    ###################
    def _channels_to_measure(self, num_parallel_supports) -> list[str]:
        process_circuit_factors = list(self._process_circuit.factorize())
        parallel_supports = [
            sorted([self.sorted_qubits.index(q) for q in p_factor.all_qubits()])
            for p_factor in process_circuit_factors
        ]
        support_sets = [
            sum(k, []) for k in itertools.combinations(parallel_supports, num_parallel_supports)
        ]

        channels = {tuple(support): [] for support in support_sets}
        for support in support_sets:
            p_channels = KNR._all_pauli_channels(len(support))
            p_channels.remove("I" * len(support))
            for chan in p_channels:
                p_str = "".join(
                    chan[support.index(k)] if k in support else "I" for k in range(self.num_qubits)
                )

                channels[tuple(support)].append(p_str)

        return channels

    @staticmethod
    def _all_pauli_channels(n):
        _p = ["I", "X", "Y", "Z"]
        return ["".join(s) for s in itertools.product(_p, repeat=n)]

    def _build_circuits(
        self,
        num_circuits: int,
        cycle_depths: list[int],
    ) -> Sequence[Sample]:
        """Build a list of random circuits to perform the XEB experiment with.

        Args:
            num_circuits: Number of circuits to generate.
            cycle_depths: An iterable of the different numbers of cycles to include in each circuit.
                For CB the provided depths are multiplied by the
                matrix root of the process circuit.

        Returns:
            The list of experiment samples.
        """
        if len(cycle_depths) != 2:
            raise ValueError("")  # Only needs two cycle depths.

        samples = []
        for channel, depth, _ in tqdm.contrib.itertools.product(
            self.pauli_channels, cycle_depths, range(num_circuits), desc="Building circuits"
        ):
            circuit, final_twirl = self._pie_circuit(channel, depth * self._matrix_root)
            samples.append(
                Sample(
                    raw_circuit=circuit,
                    data={
                        "pauli_channel": channel,
                        "cycle_depth": depth * self._matrix_root,
                        "final_twirl": final_twirl,
                    },
                )
            )
        return samples

    @staticmethod
    def _find_process_identity_min_depth(circuit, max_depth: int = 50) -> int:
        """TODO"""
        identity = np.identity(2 ** len(circuit.all_qubits()), dtype=complex)
        process_unitary = cirq.unitary(circuit)
        unitary = np.array(1 + 0j)
        for i in range(max_depth):
            unitary = np.dot(process_unitary, unitary)
            if cirq.equal_up_to_global_phase(unitary, identity):
                return i + 1
        raise RuntimeError(f"Could not find a circuit root less than {max_depth}")

    def _state_prep_circuit(self, pauli_string):
        return cirq.Circuit(
            [_STRING_TO_ROTATION[s](self.sorted_qubits[ii]) for ii, s in enumerate(pauli_string)]
        )

    def _pie_circuit(self, pauli_string, m):
        spam_circuit = self._state_prep_circuit(pauli_string)

        circuit = spam_circuit.copy()
        twirl = self._generate_n_qubit_pauli_moment()
        circuit += cirq.Moment(
            gate(qubit) for qubit, gate in twirl.items()
        )  # Avoid combined or empty gates

        for _ in range(m):
            circuit += self._process_circuit
            tc_t = twirl.after(self._process_circuit) * (
                twirl := self._generate_n_qubit_pauli_moment()
            )
            circuit += cirq.Moment(gate(qubit) for qubit, gate in tc_t.items())

        circuit += spam_circuit.copy()  # H^m = I in line 1.2 of ref.
        twirl_comp = twirl.after(spam_circuit.copy())
        circuit += cirq.Moment(gate(qubit) for qubit, gate in twirl_comp.items())

        final_twirl_string = "".join(
            _PAULI_INDEX[k] for k in twirl_comp.dense(self.qubits).pauli_mask
        )

        circuit += cirq.measure(self.sorted_qubits)

        circuit = cirq.merge_single_qubit_moments_to_phxz(
            circuit, context=cirq.TransformerContext(tags_to_ignore=("no_compile"))
        )

        return circuit, final_twirl_string

    def _generate_n_qubit_pauli_moment(
        self,
    ) -> tuple[str, cirq.Moment]:
        """Generates an n-qubit random Pauli sequence.

        Returns:
            Returns a `tuple` object of n-qubit pauli strings and their
            corresponding gate operations.
        """
        pauli_string = self._random_pauli_string()
        return cirq.DensePauliString(pauli_string).on(*self.qubits)

    def _random_pauli_string(self):
        return "".join(random.choices(list(_STRING_TO_PAULI.keys()), k=self.num_qubits))

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
            results = {"p_positive": 0.0, "p_negative": 0.0}
            for outcome, prob in sample.probabilities.items():
                if (
                    self._result_parity(
                        sample.data["pauli_channel"], sample.data["final_twirl"], outcome
                    )
                    == 0.0
                ):
                    results["p_positive"] += prob
                else:
                    results["p_negative"] += prob
            records.append(
                {
                    "pauli_channel": sample.data["pauli_channel"],
                    "cycle_depth": sample.data["cycle_depth"],
                    "final_twirl": sample.data["final_twirl"],
                    "parity": results["p_positive"] - results["p_negative"],
                    **sample.probabilities,
                }
            )
        return pd.DataFrame(records)

    def analyze_results(self, plot_results: bool = True) -> KNRResults:
        m = self.raw_data.cycle_depth.unique()
        df = (
            self.raw_data.groupby(["pauli_channel", "cycle_depth"])
            .sum()
            .reset_index()
            .pivot(columns="cycle_depth", values="parity", index="pauli_channel")
        )
        fidelities = (
            ((df[m[0]] / df[m[1]]) ** (1 / (m[0] - m[1]))).reset_index().rename_axis("", axis=1)
        )
        fidelities.rename(columns={0: "orbital_fidelity"}, inplace=True)

        marginal_probs = []
        for channel_set in self._support_channels.values():
            for c_1 in channel_set:
                p = 1.0  # 1.0 for contribution from I...I channel
                for c_2 in channel_set:
                    # Use Lemma 3 of ref. Note no renormalisation needed as process is assumed Clifford
                    p += (
                        KNR.channel_character(c_1, c_2)
                        * fidelities.query(f"pauli_channel == '{c_2}'")["orbital_fidelity"].item()
                    )
                marginal_probs.append({"pauli_channel": c_1, "marginal_probability": p})

        fidelities = pd.merge(fidelities, pd.DataFrame(marginal_probs), on="pauli_channel")
        return fidelities

    def plot_results(self) -> None:
        """Plot the experiment data and the corresponding fits."""
        pass

    def _result_parity(self, pauli_string, final_twirl, outcome):
        x, _ = KNR.bitstring_rep(final_twirl)
        s = np.fromiter(map(int, outcome), dtype=int)  # Outcome string
        xs = s  # (x + s) % 2
        s_q = np.array([s != "I" for s in pauli_string], dtype=int)  # Support of the pauli channel
        return (xs * s_q).sum() % 2

    @staticmethod
    def bitstring_rep(channel):
        ax = np.zeros(len(channel))
        az = np.zeros(len(channel))
        for idx, gate in enumerate(channel):
            if gate == "X":
                ax[idx] = 1
            if gate == "Z":
                az[idx] = 1
            if gate == "Y":
                ax[idx] = 1
                az[idx] = 1

        return ax, az

    @staticmethod
    def channel_character(base, channel):
        ax0, az0 = KNR.bitstring_rep(base)
        ax1, az1 = KNR.bitstring_rep(channel)
        return 1 - 2 * ((ax0 * az1 + az0 * ax1).sum() % 2)
