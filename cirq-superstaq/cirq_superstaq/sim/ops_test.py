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
import cirq
import numpy as np

import cirq_superstaq as css


def test_kraus_channel() -> None:
    depol = cirq.depolarize(0.1)
    gate = css.sim.KrausChannel.from_channel(depol, key="foo")
    np.testing.assert_allclose(cirq.kraus(gate), cirq.kraus(depol))
    assert cirq.measurement_key_name(gate) == "foo"

    gate = css.sim.KrausChannel.from_channel(depol, qid_shape=(3,), key="bar")
    assert cirq.measurement_key_name(gate) == "bar"

    kraus = cirq.kraus(gate)
    for i in range(4):
        assert kraus[i].shape == (3, 3)
        np.testing.assert_allclose(kraus[i][:2, :2], cirq.kraus(depol)[i])
        np.testing.assert_allclose(kraus[i][2:], 0.0)
        np.testing.assert_allclose(kraus[i][:, 2:], 0.0)

    np.testing.assert_allclose(kraus[-1], np.diag([0, 0, 1]), atol=2e-8)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(gate.on(cirq.LineQid(0, 3))),
        "0 (d=3): ───Kraus(3)───",
    )


def test_jump_channel() -> None:
    transition_matrix = [
        [0.0, 0.01, 0.0],
        [0.0, 0.04, 0.0],
        [0.0, 0.0, 0.09],
        [0.01, 0.0, 0.0],
    ]
    gate = css.sim.JumpChannel(transition_matrix)
    np.testing.assert_array_equal(
        gate._transition_matrix_(),
        [
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.09, 0.0],
            [0.01, 0.0, 0.0, 0.0],
        ],
    )

    assert gate == css.sim.JumpChannel(gate._transition_matrix_())
    assert cirq.qid_shape(gate) == (2,)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gate.on(cirq.q(0))),
        "0: ───Jump(4)───",
    )

    kraus_chan = gate.to_kraus_channel(4, key="foo")
    assert cirq.measurement_key_name(kraus_chan) == "foo"
    np.testing.assert_array_equal(
        cirq.kraus(kraus_chan),
        [
            [
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            np.diag(np.sqrt(1 - np.array([0.01, 0.05, 0.09, 0.0]))).tolist(),
        ],
    )


def test_jump_channel_then() -> None:
    rng = np.random.default_rng()
    channel_1 = css.sim.JumpChannel(rng.uniform(0, 0.01, (4, 4)))
    channel_2 = css.sim.JumpChannel(rng.uniform(0, 0.01, (3, 2)))

    initial_state = cirq.testing.random_density_matrix(5)
    q0 = cirq.LineQid(0, 5)

    # channel_1 followed by channel_2:
    np.testing.assert_allclose(
        cirq.final_density_matrix(
            channel_1.then(channel_2).to_kraus_channel(dimension=5),
            initial_state=initial_state,
            dtype=np.complex128,
        ),
        cirq.final_density_matrix(
            cirq.Circuit(
                channel_1.to_kraus_channel(dimension=5).on(q0),
                channel_2.to_kraus_channel(dimension=5).on(q0),
            ),
            initial_state=initial_state,
            dtype=np.complex128,
        ),
        atol=1e-6,
    )

    # channel_2 followed by channel_1:
    np.testing.assert_allclose(
        cirq.final_density_matrix(
            channel_2.then(channel_1).to_kraus_channel(dimension=5),
            initial_state=initial_state,
            dtype=np.complex128,
        ),
        cirq.final_density_matrix(
            cirq.Circuit(
                channel_2.to_kraus_channel(dimension=5).on(q0),
                channel_1.to_kraus_channel(dimension=5).on(q0),
            ),
            initial_state=initial_state,
            dtype=np.complex128,
        ),
        atol=1e-6,
    )


def test_qudit_subspace_gate() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    gate = css.sim.QuditPermutationGate([1, 2, 0], 2)
    op = gate.on(q0, q1, q2)
    cirq_op = cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2)

    np.testing.assert_array_equal(cirq.unitary(op), cirq.unitary(cirq_op))
    cirq.testing.assert_same_circuits(
        cirq.Circuit(cirq.decompose_once(op)),
        cirq.Circuit(cirq.SWAP(q0, q1), cirq.SWAP(q0, q2)),
    )
    assert css.sim.ops.qudit_permutation_op([1, 2, 0], q0, q1, q2) == cirq_op

    t0, t1, t2 = cirq.LineQid.range(3, dimension=3)
    op = css.sim.ops.qudit_permutation_op([1, 2, 0], t0, t1, t2)
    assert op == css.sim.QuditPermutationGate([1, 2, 0], 3).on(t0, t1, t2)

    np.testing.assert_array_equal(
        cirq.unitary(op),
        cirq.Circuit(css.SWAP3(t0, t1), css.SWAP3(t0, t2)).unitary(),
    )
    cirq.testing.assert_same_circuits(
        cirq.Circuit(cirq.decompose_once(op)),
        cirq.Circuit(css.SWAP3(t0, t1), css.SWAP3(t0, t2)),
    )


def test_with_dimension() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    transition_matrix = np.array([[0.01, 0.04, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]])
    circuit = cirq.Circuit(
        cirq.X(q0),
        css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.amplitude_damp(0.1).on(q0),
        css.barrier(q0, q1, q2),
        css.sim.JumpChannel(transition_matrix).on(q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.measure(q0, q1, q2, key="key"),
    )

    cirq.testing.assert_same_circuits(
        css.sim.with_dimension(circuit, dimension=2),
        circuit,
    )

    t0, t1, t2 = cirq.LineQid.range(3, dimension=3)
    expected = cirq.Circuit(
        css.qubit_subspace_op(cirq.X(q0), [3]),
        css.qubit_subspace_op(css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2), [3, 3, 3]),
        css.sim.KrausChannel.from_channel(cirq.amplitude_damp(0.1), qid_shape=(3,)).on(t0),
        css.barrier(t0, t1, t2),
        css.sim.JumpChannel(transition_matrix).to_kraus_channel(dimension=3).on(t2),
        css.sim.QuditPermutationGate([1, 2, 0], 3).on(t0, t1, t2),
        cirq.measure(t0, t1, t2, key="key"),
    )
    cirq.testing.assert_same_circuits(
        css.sim.with_dimension(circuit, dimension=3),
        expected,
    )

    qq0, qq1, qq2 = cirq.LineQid.range(3, dimension=4)
    expected = cirq.Circuit(
        css.qubit_subspace_op(cirq.X(q0), [4]),
        css.qubit_subspace_op(css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2), [4, 4, 4]),
        css.sim.KrausChannel.from_channel(cirq.amplitude_damp(0.1), qid_shape=(4,)).on(qq0),
        css.barrier(qq0, qq1, qq2),
        css.sim.JumpChannel(transition_matrix).to_kraus_channel(dimension=4).on(qq2),
        css.sim.QuditPermutationGate([1, 2, 0], 4).on(qq0, qq1, qq2),
        cirq.measure(qq0, qq1, qq2, key="key"),
    )
    cirq.testing.assert_same_circuits(
        css.sim.with_dimension(circuit, dimension=4),
        expected,
    )
