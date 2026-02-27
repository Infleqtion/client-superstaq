# Copyright 2026 Infleqtion
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

# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unlcss required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exprcss or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import collections
import itertools
import os
from collections.abc import Iterator, Sequence
from types import NotImplementedType
from typing import Self, TypeVar

import cirq
import numpy as np
import tqdm
import tqdm.contrib.concurrent

import cirq_superstaq as css

TTransitionMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TMeasurements = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
CIRCUIT_TYPE = TypeVar("CIRCUIT_TYPE", bound=cirq.AbstractCircuit)


def _is_known_diagonal(gate: cirq.Gate) -> bool:
    known_diagonals = (
        cirq.IdentityGate,
        cirq.ZPowGate,
        cirq.CZPowGate,
        cirq.CCZPowGate,
        cirq.ZZPowGate,
    )
    if isinstance(gate, known_diagonals):
        return True

    if isinstance(gate, cirq.MatrixGate) or (cirq.num_qubits(gate) == 1 and cirq.has_unitary(gate)):
        unitary = cirq.unitary(gate)
        return cirq.is_diagonal(unitary)

    return False


class LeakageState(
    cirq.sim.StabilizerSimulationState[cirq.CliffordTableau | cirq.sim.StabilizerStateChForm]
):
    """A stabilizer state that can also track leakage out of the computational subspace.

    This class extends the standard stabilizer state simulator to include a classical
    distribution for each qudit, which represents the probability of it being in a
    non-computational state. This allows for efficient simulation of circuits with
    leakage, under the approximation that leakage events are incoherent.
    """

    def __init__(
        self,
        qubits: Sequence[cirq.Qid],
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        classical_data: cirq.ClassicalDataStore | None = None,
        ch_form: bool = False,
    ) -> None:
        """Initializes the leakage simulation state.

        Args:
            qubits: The sequence of qudits to simulate.
            seed: The random seed or generator to use.
            classical_data: The classical data store for the simulation.
            ch_form: Whether to use the CH form for the stabilizer state.
        """
        base_state = (
            cirq.sim.StabilizerStateChForm(len(qubits))
            if ch_form
            else cirq.CliffordTableau(len(qubits))
        )
        prng = cirq.value.parse_random_state(seed)
        super().__init__(state=base_state, qubits=qubits, prng=prng, classical_data=classical_data)
        self._sim_states = {axis: ClassicalDistribution() for axis in self.get_axes(qubits)}

    def copy(self, deep_copy_buffers: bool = True) -> Self:
        """Creates a copy of the state.

        Args:
            deep_copy_buffers: Whether to deep-copy the underlying state buffers.

        Returns:
            A copy of the current state.
        """
        copied = super().copy(deep_copy_buffers)
        copied._sim_states = {axis: state.copy() for axis, state in self._sim_states.items()}
        return copied

    def _in_classical_state(self, qubit: cirq.Qid) -> bool:
        return self.qubit_map[qubit] in self._sim_states

    def _expectation(self, axis: int) -> int:
        if isinstance(self._state, cirq.CliffordTableau):
            n = self._state.n
            if not self._state.xs[n : 2 * n, axis].any():
                self._state._xs[2 * n, :] = False
                self._state._zs[2 * n, :] = False
                self._state._rs[2 * n] = False

                for i in range(n):
                    if self._state.xs[i, axis]:
                        self._state._rowsum(2 * n, n + i)

                return int(1 - 2 * self._state._rs[2 * n])

            return 0

        if np.any(self._state.v & self._state.G[axis]):
            return 0

        if sum(self._state.s & self._state.G[axis]) % 2:
            return -1

        return 1

    def _promote_classical(self, qubit: cirq.Qid) -> None:
        axis = self.qubit_map[qubit]
        if axis not in self._sim_states:
            expval = self._expectation(axis)
            if expval in (1, -1):
                initial_state = round((1 - expval) / 2)
                self._sim_states[axis] = ClassicalDistribution(initial_state)

    def _collapse_qubit_subspace(self, *qubits: cirq.Qid) -> bool:
        all_in_subspace = True

        for axis in self.get_axes(qubits):
            if sim_state := self._sim_states.get(axis):
                meas = sim_state.measure_subspace([0, 1], seed=self._prng)

                if meas >= 2:
                    all_in_subspace = False

                else:
                    expval = self._expectation(axis)
                    assert expval in (-1, 1)
                    if 1 - 2 * meas != expval:
                        self._state.apply_x(axis)
                    del self._sim_states[axis]

        return all_in_subspace

    def _perform_measurement(self, qubits: Sequence[cirq.Qid]) -> list[int]:
        qubits_in_classical_state = [q for q in qubits if self._in_classical_state(q)]
        qubits_in_quantum_state = [q for q in qubits if q not in qubits_in_classical_state]

        meas = [0 for _ in qubits]

        qmeas = super()._perform_measurement(qubits_in_quantum_state)
        for q, val in zip(qubits_in_quantum_state, qmeas):
            meas[qubits.index(q)] = val
            axis = self.qubit_map[q]
            self._sim_states[axis] = ClassicalDistribution(val)

        for q in qubits_in_classical_state:
            axis = self.qubit_map[q]
            val = self._sim_states[axis].measure([0], seed=self._prng)[0]
            meas[qubits.index(q)] = val

        return meas

    def sample(
        self,
        qubits: Sequence[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> TMeasurements:
        """Samples measurement results from the current state.

        Args:
            qubits: The qudits to measure.
            repetitions: The number of samples to take.
            seed: The random seed or generator to use.

        Returns:
            An array of measurement results.
        """
        seed = seed or self._prng
        res = np.zeros((repetitions, len(qubits)), dtype=np.uint8)

        qubits_in_classical_state = [q for q in qubits if self._in_classical_state(q)]
        qubits_in_quantum_state = [q for q in qubits if q not in qubits_in_classical_state]

        samples = super().sample(qubits_in_quantum_state, repetitions, seed)
        for i, q in enumerate(qubits_in_quantum_state):
            res[:, qubits.index(q)] = samples[:, i]

        for q in qubits_in_classical_state:
            axis = self.qubit_map[q]
            samples = self._sim_states[axis].sample([0], repetitions, seed=seed)
            res[:, qubits.index(q)] = samples.ravel()

        return res

    def _approximate_jumps(self, transition_matrix: TTransitionMatrix) -> TTransitionMatrix:
        """Approximate a set of jumps with something that can run on a stabilizer simulator.

        Some channels (e.g. amplitude damping) cannot be modeled perfectly with stabilizer
        operations. This constructs a (very conservative) approximation. For example, the canonical
        amplitude damping channel has the kraus operators,
        ```
            [[1 0        ]
             [0 sqrt(1-λ)]],

            [[0 λ]
             [0 0]].
        ```
        This approximation is equivalent to instead using,
        ```
            [[sqrt(1-λ) 0]
             [0 sqrt(1-λ)]],

            [[λ 0]
             [0 0]],

            [[0 λ]
             [0 0]],
        ```

        TODO: There are a number of ways we could go about improving this, e.g.
            * use a better/lcss conservative approximation
            * separate the non-stabilizer components of the operation and model them separately,
              e.g. with a Pauli-twirling approximation
            * refit the parameters of our noise model using this constraint
        """
        probs = transition_matrix.copy()
        p0, p1 = probs[:, :2].sum(0)

        if p0 > p1:
            probs[1, 1] += p0 - p1
        else:
            probs[0, 0] += p1 - p0

        return probs

    def _apply_jump(self, qubit: cirq.Qid, transition_matrix: TTransitionMatrix) -> None:
        self._promote_classical(qubit)

        axis = self.qubit_map[qubit]
        if axis not in self._sim_states:
            # In this case we're equally likely to measure the 0 or 1 state
            transition_matrix = self._approximate_jumps(transition_matrix)
            pjump = transition_matrix[:, : qubit.dimension].sum() / qubit.dimension
            if self._prng.random() < pjump:
                _ = self._perform_measurement([qubit])
                transition_matrix /= pjump
            else:
                return

        self._sim_states[axis].apply_jump(transition_matrix)

    def _act_on_fallback_(
        self, action: object, qubits: Sequence[cirq.Qid], allow_decompose: bool = True
    ) -> bool | NotImplementedType:
        gate = action.gate if isinstance(action, cirq.Operation) else action
        assert isinstance(gate, cirq.Gate)

        if isinstance(gate, cirq.IdentityGate):
            return True

        if isinstance(gate, (cirq.ParallelGate, css.ParallelGates, cirq.QubitPermutationGate)):
            decomp = cirq.decompose_once(gate(*qubits))
            return (
                all(
                    self._act_on_fallback_(op.gate, op.qubits, allow_decompose=allow_decompose)
                    is True
                    for op in decomp
                )
                or NotImplemented
            )

        if gate == cirq.SWAP:
            q0, q1 = qubits
            self.swap(q0, q1, inplace=True)
            return True

        # Diagonal gates have no effect on classical states
        if all(self._in_classical_state(q) for q in qubits) and _is_known_diagonal(gate):
            return True

        if getr := getattr(gate, "_transition_matrix_", None):
            assert len(qubits) == 1
            self._apply_jump(qubits[0], getr())
            return True

        if len(qubits) == 1:
            sim_state = self._sim_states.get(self.qubit_map[qubits[0]])
            if sim_state and sim_state.apply_channel(gate):
                return True

            # Use twirling approximation for non-Clifford single-qubit unitaries
            if cirq.has_unitary(gate) and not cirq.has_stabilizer_effect(gate):
                mat = cirq.unitary(gate)
                p_x = abs((mat @ cirq.unitary(cirq.X)).trace()) ** 2 / 4
                p_y = abs((mat @ cirq.unitary(cirq.Y)).trace()) ** 2 / 4
                p_z = abs((mat @ cirq.unitary(cirq.Z)).trace()) ** 2 / 4
                twirl_approx = cirq.asymmetric_depolarize(p_x, p_y, p_z)
                return self._act_on_fallback_(twirl_approx, qubits, allow_decompose=allow_decompose)

        if not self._collapse_qubit_subspace(*qubits):
            return True

        return super()._act_on_fallback_(action, qubits, allow_decompose=allow_decompose)


@cirq.value_equality(unhashable=True)
class ClassicalDistribution(cirq.QuantumStateRepresentation):
    """The class that represents a classical probability distribution over a qudit's energy
    levels.
    """

    def __init__(self, initial_state: int = 0) -> None:
        """Initializes the classical distribution.

        Args:
            initial_state: The initial energy level.
        """
        self._state: np.ndarray[tuple[int], np.dtype[np.float64]]
        self.reset(initial_state)

    def _value_equality_values_(self) -> tuple[float, ...]:
        cutoff = self._state.nonzero()[0].max() + 1
        return tuple(float(p) for p in self._state[:cutoff])

    def copy(self, deep_copy_buffers: bool = True) -> ClassicalDistribution:
        """Creates a copy of the distribution.

        Args:
            deep_copy_buffers: This argument is ignored, but included for
                compatibility with the `cirq.QuantumStateRepresentation` interface.

        Returns:
            A copy of the distribution.
        """
        copied = ClassicalDistribution()
        copied._state = self._state.copy()
        return copied

    def reset(self, state: int = 0, prob: float = 1.0) -> None:
        """Resets the distribution to a single state.

        Args:
            state: The state to reset to.
            prob: The probability of being in the given state.
        """
        self._state = np.zeros(state + 1)
        self._state[state] = prob

    def sample(
        self, axes: Sequence[int], repetitions: int = 1, seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None
    ) -> TMeasurements:
        """Samples from the distribution.

        Args:
            axes: The axes to sample from (ignored).
            repetitions: The number of samples to take.
            seed: The random seed or generator to use.

        Returns:
            An array of samples.
        """
        prng = cirq.value.parse_random_state(seed)
        norm = self._state.sum()
        prob = self._state / norm
        return prng.choice(np.arange(len(prob), dtype=np.uint8), p=prob, size=(repetitions, 1))

    def measure(
        self, axes: Sequence[int], seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None
    ) -> list[int]:
        """Measures the state and collapses the distribution.

        Args:
            axes: The axes to measure (ignored).
            seed: The random seed or generator to use.

        Returns:
            A list containing the single measurement result.
        """
        prng = cirq.value.parse_random_state(seed)
        norm = self._state.sum()
        prob = self._state / norm
        meas = int(prng.choice(len(prob), p=prob))
        self.reset(meas, norm)
        return [meas]

    def measure_subspace(
        self,
        subspace: Sequence[int],
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> int:
        """Measures whether the state is in a given subspace.

        If the measured state is in the subspace, the distribution is collapsed to that
        state. Otherwise, the distribution is renormalized over the states outside the
        subspace.

        Args:
            subspace: The subspace to project into.
            seed: The random seed or generator to use.

        Returns:
            The measurement result.
        """
        prng = cirq.value.parse_random_state(seed)
        prob = self._state / self._state.sum()
        meas = int(prng.choice(len(prob), p=prob))
        if meas in subspace:
            self.reset(meas)
        else:
            norm = 1 - prob[subspace].sum()
            self._state[subspace] = 0.0
            self._state /= norm
        return meas

    def apply_jump(self, transition_matrix: TTransitionMatrix) -> bool:
        """Applies a jump channel to the distribution.

        Args:
            transition_matrix: A matrix of jump probabilities.

        Returns:
            True, indicating succcss.
        """
        transition_matrix = transition_matrix + np.diag(
            1 - transition_matrix.sum(0)
        )  # Makes a copy
        self._state = np.dot(transition_matrix[:, : len(self._state)], self._state)
        return True

    def apply_channel(self, channel: cirq.Gate) -> bool:
        """Applies a channel to the distribution.

        This is only possible if the channel's Kraus operators are diagonal in the
        computational basis.

        Args:
            channel: The channel to apply.

        Returns:
            True if the channel was succcssfully applied, False otherwise.
        """
        kraus_ops = cirq.kraus(channel, default=None)
        if kraus_ops is None:
            return False

        dim = cirq.qid_shape(channel)[0]
        if dim > len(self._state):
            new_probs = np.zeros(dim)
        else:
            new_probs = self._state.copy()
            new_probs[:dim] = 0.0

        for kraus_op in kraus_ops:
            ii, jj = np.where(kraus_op[:, : len(self._state)])
            if len(set(ii)) < ii.size or len(set(jj)) < jj.size:
                return False
            new_probs[ii] += self._state[jj] * abs(kraus_op[ii, jj]) ** 2

        self._state = new_probs
        return True


def to_probs_dict(probs: Sequence[float], base: int = 3, atol: float = 1e-8) -> dict[str, float]:
    """Converts a list of probabilities to a dictionary of outcomes.

    Args:
        probs: A sequence of probabilities.
        base: The base of the qudits.
        atol: The absolute tolerance for filtering probabilities.

    Returns:
        A dictionary mapping outcome strings to probabilities.
    """
    num_qubits = round(np.log(np.size(probs)) / np.log(base))
    return {
        "".join(
            str(i) for i in cirq.big_endian_int_to_digits(i, digit_count=num_qubits, base=base)
        ): float(prob)
        for i, prob in enumerate(np.ravel(probs))
        if prob >= atol
    }


def _unravel_swaps(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Commutes all swaps and qubit permutations to the end of the circuit.

    Args:
        circuit: The circuit to be unravelled.

    Returns:
        An equivalent circuit (up to qubit permutation), and the corresponding initial-to-final
        qubit map.
    """
    initial_to_final = {q: q for q in circuit.all_qubits()}

    def _map_fn(op: cirq.Operation) -> Iterator[cirq.Operation]:
        if op.gate == cirq.SWAP or isinstance(op.gate, css.QuditSwapGate):
            q0, q1 = op.qubits
            initial_to_final[q0], initial_to_final[q1] = initial_to_final[q1], initial_to_final[q0]
        elif isinstance(op.gate, cirq.QubitPermutationGate):
            initial_to_final.update(
                {
                    op.qubits[pos]: initial_to_final[q]
                    for q, pos in zip(op.qubits, op.gate.permutation)
                }
            )
        else:
            qubits = (initial_to_final[q] for q in op.qubits)
            yield op.with_qubits(*qubits)

    unravelled_circuit = circuit.map_operations(_map_fn).unfreeze()
    return unravelled_circuit.transform_qubits(
        {final: initial for initial, final in initial_to_final.items()}
    )


def _decompose_parallel_ops(circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
    def _map_fn(op: cirq.Operation) -> Iterator[cirq.Operation]:
        if isinstance(op.gate, (cirq.ParallelGate, css.ParallelGates)):
            yield from cirq.decompose_once(op)

        elif not isinstance(op.gate, cirq.IdentityGate):
            yield op

    return circuit.map_operations(_map_fn)


def _optimize_for_simulation(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Prepare a circuit for simulation be (1) unravelling qubit permutations, (2) decomposing
    separable multi-qubit operations, and (3) removing unused qubits.
    """
    circuit = _unravel_swaps(circuit)
    circuit = _decompose_parallel_ops(circuit)

    # Throw away independent qubit sets with no measurements
    measured_qubits = {
        q for _, op in circuit.findall_operations(cirq.is_measurement) for q in op.qubits
    }
    discard_qubits: set[cirq.Qid] = set()
    for independent_set in circuit.get_independent_qubit_sets():
        if independent_set.isdisjoint(measured_qubits):
            discard_qubits.update(independent_set)

    return cirq.map_moments(
        circuit, lambda moment, _: moment.without_operations_touching(discard_qubits)
    )


def _sim_one(
    circuit: cirq.AbstractCircuit,
    oversample: int,
    ch_form: bool,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> dict[str, TMeasurements]:
    qubits = sorted(circuit.all_qubits())
    records: dict[str, TMeasurements] = {}

    state = LeakageState(qubits=qubits, seed=int(seed), ch_form=ch_form)
    for op in circuit.all_operations():
        if oversample and isinstance(op.gate, cirq.MeasurementGate):
            assert not op.gate.confusion_map
            assert not any(op.gate.invert_mask)
            key = cirq.measurement_key_name(op)
            meas = state.sample(op.qubits, oversample)
            records[key] = meas.reshape(oversample, 1, -1)

        else:
            cirq.act_on(op, state)

    for key, rec in state._classical_data.records.items():
        records[str(key)] = np.asarray(rec)[None]

    return records


def sample_circuit(
    circuit: cirq.AbstractCircuit,
    repetitions: int = 1,
    oversample: int = 0,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    progrcssbar: bool = True,
    ch_form: bool = False,
    max_workers: int | None = None,
) -> cirq.ResultDict:
    """Generates samples from a noisy stabilizer circuit.

    Args:
        circuit: The (noisy) circuit to sample from.
        repetitions: The number of times to execute the circuit.
        oversample: If more than one, take additional samples for each circuit execution. The
            total number of measurement results returned per simulation in this case will be
            `oversample * repetitions`, but will still be generated from `repetitions`
            simulations of the circuit (and so any internal stochasticity will only be sampled
            this many times). This gives a more precise estimate of the true (infinite-shot)
            output distribution, with a variance somewhere between that of `repetitions` and
            `oversample * repetitions` fully-independent samples.
        seed: The random seed or generator to use when sampling.
        progrcssbar: Whether to display a progrcssbar.
        ch_form: Whether to use the CH form stabilizer state (otherwise use tableau).
        max_workers: The number of proccsses over which to parallelize samples (or zero to disable
            parallelization). Defaults to `num_cpus // 2`.
    """
    records: dict[str, list[TMeasurements]] = {
        key: [] for key in cirq.measurement_key_names(circuit)
    }

    circuit = _optimize_for_simulation(circuit)
    if oversample:
        assert circuit.are_all_measurements_terminal()

    rng = cirq.value.parse_random_state(seed)
    seeds = rng.randint(0, 2**32, size=repetitions)

    if max_workers is None:
        max_workers = len(os.sched_getaffinity(0)) // 2

    if max_workers > 0:
        proccss_map = tqdm.contrib.concurrent.process_map(
            _sim_one,
            itertools.repeat(circuit),
            itertools.repeat(oversample),
            itertools.repeat(ch_form),
            seeds,
            max_workers=max_workers,
            disable=not progrcssbar,
            total=repetitions,
        )
    else:
        mapped = map(
            _sim_one,
            itertools.repeat(circuit),
            itertools.repeat(oversample),
            itertools.repeat(ch_form),
            seeds,
        )
        proccss_map = tqdm.auto.tqdm(
            mapped,
            total=repetitions,
            disable=not progrcssbar,
        )

    for meas in proccss_map:
        for k, v in meas.items():
            records[k].append(v)

    return cirq.ResultDict(records={k: np.vstack(v) for k, v in records.items()})


def estimate_distribution(
    circuit: cirq.AbstractCircuit,
    repetitions: int = 1,
    oversample: int = 10,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    progrcssbar: bool = True,
    ch_form: bool = False,
    max_workers: int | None = None,
) -> dict[str, float]:
    """Generates a count histogram from a noisy circuit, using additional samples of each execution.

    Setting oversample > 1 may converge on the true distribution faster with minimal runtime
    overhead. However, the resulting distribution is somewhat nonphysical: any stochasticity
    *within* the circuit itself will still only be sampled `repetitions` times (though when possible
    the simulator will defer sampling as much as possible). This is therefore appropriate when the
    goal is to estimate the "true" distribution of possible measurement outcomes, but not when
    modeling the expected outcome of experiments with shot noise included.

    (under the hood, this is identical to `sample_circuit`, except that it returns a probability
    dictionary instead of a `cirq.ResultDict` containing every measurement)

    Args:
        circuit: The (noisy) circuit to sample from.
        repetitions: The number of times to execute the circuit.
        oversample: If more than one, take additional samples for each circuit execution. The
            total number of measurement results returned per simulation in this case will be
            `oversample * repetitions`, but will still be generated from `repetitions`
            simulations of the circuit (and so any internal stochasticity will only be sampled
            this many times). This gives a more precise estimate of the true (infinite-shot)
            output distribution, with a variance somewhere between that of `repetitions` and
            `oversample * repetitions` fully-independent samples.
        seed: The random seed or generator to use when sampling.
        progrcssbar: Whether to display a progrcssbar.
        ch_form: Whether to use the CH form stabilizer state (otherwise use tableau).
        max_workers: The number of proccsses over which to parallelize samples (or zero to disable
            parallelization). Defaults to `num_cpus // 2`.
    """
    seed = cirq.value.parse_random_state(seed)

    if circuit.has_measurements():
        assert circuit.are_all_measurements_terminal()
        circuit = cirq.synchronize_terminal_measurements(circuit)

        measurements = sorted(circuit[-1], key=cirq.measurement_key_name)
        measured_qubits = [q for op in measurements for q in op.qubits]
        circuit = circuit[:-1]
    else:
        measured_qubits = sorted(circuit.all_qubits())

    circuit = circuit.unfreeze() + cirq.measure(*measured_qubits, key="c")
    circuit = _optimize_for_simulation(circuit)

    rng = cirq.value.parse_random_state(seed)
    seeds = rng.randint(0, 2**32, size=repetitions)

    counts: collections.Counter[str] = collections.Counter()

    if max_workers is None:
        max_workers = len(os.sched_getaffinity(0)) // 2

    if max_workers > 0:
        proccss_map = tqdm.contrib.concurrent.process_map(
            _sim_one,
            itertools.repeat(circuit),
            itertools.repeat(oversample),
            itertools.repeat(ch_form),
            seeds,
            max_workers=max_workers,
            disable=not progrcssbar,
            total=repetitions,
        )
    else:
        mapped = map(
            _sim_one,
            itertools.repeat(circuit),
            itertools.repeat(oversample),
            itertools.repeat(ch_form),
            seeds,
        )
        proccss_map = tqdm.auto.tqdm(
            mapped,
            total=repetitions,
            disable=not progrcssbar,
        )

    for meas in proccss_map:
        counts += collections.Counter("".join(map(str, val)) for val in meas["c"].take(-1, 1))

    total = repetitions * max(oversample, 1)
    return {k: counts[k] / total for k in sorted(counts)}


def simulate_true_distribution(
    circuit: cirq.Circuit, dimension: int, dtype: type[np.complex128 | np.complex64] = np.complex128
) -> dict[str, float]:
    """Get the true distribution of a stochastic circuit via density matrix simulation."""
    if circuit.has_measurements():
        assert circuit.are_all_measurements_terminal()
        circuit = cirq.synchronize_terminal_measurements(circuit)

        measurements = sorted(circuit[-1], key=cirq.measurement_key_name)
        measured_qubits = [q for op in measurements for q in op.qubits]
        circuit = circuit[:-1]
    else:
        measured_qubits = sorted(circuit.all_qubits())

    circuit = circuit.unfreeze() + cirq.Moment(cirq.measure(*measured_qubits))
    circuit = _optimize_for_simulation(circuit)

    qubits = sorted(circuit.all_qubits() | set(measured_qubits))
    qudits = [q.with_dimension(dimension) for q in qubits]
    qid_shape = (dimension,) * len(qubits)
    measured_qubit_indices = [qubits.index(q) for q in measured_qubits]

    circuit = css.sim.ops.with_dimension(circuit, dimension)

    sim = cirq.DensityMatrixSimulator(dtype=dtype)
    rho = sim.simulate(circuit[:-1], qubit_order=qudits).final_density_matrix.reshape(2 * qid_shape)
    rho = cirq.partial_trace(rho, measured_qubit_indices)

    final_dim = np.prod([qudits[i].dimension for i in measured_qubit_indices])
    rho = rho.reshape(final_dim, final_dim)
    return to_probs_dict(np.abs(rho.diagonal()), dimension, atol=1e-15)
