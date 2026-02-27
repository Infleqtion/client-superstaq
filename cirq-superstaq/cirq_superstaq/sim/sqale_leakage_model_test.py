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
from __future__ import annotations

from collections.abc import Mapping

import cirq
import cirq_superstaq as css
import numpy as np

import experiments_superstaq as ess


def _assert_distributions_approx_eq(
    dist0: Mapping[str, float],
    dist1: Mapping[str, float],
    atol: float = 1e-8,
) -> None:
    assert all(
        np.isclose(dist0.get(key, 0), dist1.get(key, 0), atol=atol)
        for key in dist0.keys() | dist1.keys()
    ), (dist0, dist1)


def test_scale_by() -> None:
    params1 = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS
    params2 = params1.scale_by(2.0)
    params3 = params2.scale_by(3.0)

    for key, val in params3.model_dump().items():
        orig_val = params1.model_dump()[key]

        if key in (
            "movement_phase_error",
            "rz_relative_overrotation",
            "cz_phase_error",
        ):
            assert val == 6 * orig_val
        else:
            assert val == orig_val


def test_spam_errors() -> None:
    q0, q1 = cirq.q(3), cirq.q(1)
    circuit = cirq.Circuit(
        cirq.X(q1),
        cirq.measure(q0, q1),
    )
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit, 5)
    assert res == {"01": 1.0}

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel()
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    assert res == {"01": 1.0}

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        initial_state_probs=[0.22, 0.7, 0.03, 0.0, 0.05]
    )
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "00": 0.73 * 0.25,
            "01": 0.73 * 0.7,
            "02": 0.73 * 0.05,
            "10": 0.22 * 0.25,
            "11": 0.22 * 0.7,
            "12": 0.22 * 0.05,
            "20": 0.05 * 0.25,
            "21": 0.05 * 0.7,
            "22": 0.05 * 0.05,
        },
    )

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(classifier_errors=(0.01, 0.1))
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "00": 0.99 * 0.1,
            "01": 0.99 * 0.9,
            "10": 0.01 * 0.1,
            "11": 0.01 * 0.9,
        },
    )


def test_rz_errors() -> None:
    q0, q1 = cirq.q(0), cirq.q(1)
    circuit = cirq.Circuit(
        cirq.X(q1),
        cirq.S(q0),
        cirq.Z(q1),
    )

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel()
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    assert res == {"01": 1.0}

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        rz_transition_matrix=[
            [0, 3e-1, 0, 0],
            [1e-1, 0, 0, 0],
            [1e-2, 3e-2, 0, 0],
            [1e-3, 3e-3, 0, 0],
        ],
    )
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "00": 0.889 * 3e-1,
            "01": 0.889 * 0.667,
            "02": 0.889 * 3e-2,
            "03": 0.889 * 3e-3,
            "10": 1e-1 * 3e-1,
            "11": 1e-1 * 0.667,
            "12": 1e-1 * 3e-2,
            "13": 1e-1 * 3e-3,
            "20": 1e-2 * 3e-1,
            "21": 1e-2 * 0.667,
            "22": 1e-2 * 3e-2,
            "23": 1e-2 * 3e-3,
            "30": 1e-3 * 3e-1,
            "31": 1e-3 * 0.667,
            "32": 1e-3 * 3e-2,
            "33": 1e-3 * 3e-3,
        },
    )

    circuit = cirq.Circuit(
        cirq.H.on_each(q0, q1),
        cirq.Z(q0),
        cirq.S(q1),
        cirq.S(q1),
        cirq.H.on_each(q0, q1),
    )
    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(rz_relative_overrotation=0.1)
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "00": np.sin(0.1 * np.pi / 2) ** 4,
            "01": np.sin(0.1 * np.pi / 2) ** 2 * np.cos(0.1 * np.pi / 2) ** 2,
            "10": np.cos(0.1 * np.pi / 2) ** 2 * np.sin(0.1 * np.pi / 2) ** 2,
            "11": np.cos(0.1 * np.pi / 2) ** 4,
        },
    )


def test_gr_errors() -> None:
    q0, q1 = cirq.q(0), cirq.q(1)
    circuit = cirq.Circuit(
        css.ParallelRGate(1.0, 2.0, 2).on(q0, q1),
        css.ParallelRGate(-2.0, -1.0, 2).on(q0, q1),
        css.ParallelRGate(2 * np.pi - 1.0, np.pi + 1.0, 2).on(q0, q1),
    )

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        gr_relative_overrotation=0.25,
        gr_static_overrotation_rads=0.005,
    )
    cirq.testing.assert_same_circuits(
        circuit.with_noise(model)[-6:],
        cirq.Circuit(
            css.ParallelRGate(1.0, 2.0, 2).on(q0, q1),
            css.ParallelRGate(0.255, 2.0, 2).on(q0, q1),
            css.ParallelRGate(2.0, np.pi - 1.0, 2).on(q0, q1),
            css.ParallelRGate(0.505, np.pi - 1.0, 2).on(q0, q1),
            css.ParallelRGate(1.0, 1.0, 2).on(q0, q1),
            css.ParallelRGate(0.255, 1.0, 2).on(q0, q1),
        ),
    )


def test_cz_errors() -> None:
    q0, q1 = cirq.q(0), cirq.q(1)
    circuit = cirq.Circuit(cirq.X(q1), cirq.CZ(q0, q1))

    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit, 5)
    assert res == {"01": 1.0}

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel()
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    assert res == {"01": 1.0}

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        cz_transition_matrix=[
            [0, 3e-1, 0, 0],
            [1e-1, 0, 0, 0],
            [1e-2, 3e-2, 0, 0],
            [1e-3, 3e-3, 0, 0],
        ],
    )
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "00": 0.889 * 3e-1,
            "01": 0.889 * 0.667,
            "02": 0.889 * 3e-2,
            "03": 0.889 * 3e-3,
            "10": 1e-1 * 3e-1,
            "11": 1e-1 * 0.667,
            "12": 1e-1 * 3e-2,
            "13": 1e-1 * 3e-3,
            "20": 1e-2 * 3e-1,
            "21": 1e-2 * 0.667,
            "22": 1e-2 * 3e-2,
            "23": 1e-2 * 3e-3,
            "30": 1e-3 * 3e-1,
            "31": 1e-3 * 0.667,
            "32": 1e-3 * 3e-2,
            "33": 1e-3 * 3e-3,
        },
    )

    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.H(q1),
    )
    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(cz_phase_error=0.1)
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(res, {"10": 0.1, "11": 0.9})

    # Check correct merging of phase error into transition matrix
    rng = np.random.default_rng()
    random_transitions = rng.uniform(0, 0.01, (5, 4))
    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        cz_transition_matrix=random_transitions.tolist(),
        cz_phase_error=0.1,
    )
    circuit = cirq.Circuit(
        cirq.MatrixGate(cirq.testing.random_unitary(4)).on(q0, q1),
        cirq.CZ(q0, q1),
        cirq.MatrixGate(cirq.testing.random_unitary(4)).on(q0, q1),
        cirq.CZ(q0, q1),
        cirq.MatrixGate(cirq.testing.random_unitary(4)).on(q0, q1),
    )
    res1 = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
        cz_transition_matrix=random_transitions.tolist(),
        cz_phase_error=0.0,
    )
    circuit = cirq.Circuit(
        circuit[0],
        cirq.CZ(q0, q1),
        cirq.phase_flip(0.1).on_each(q0, q1),
        circuit[2],
        cirq.CZ(q0, q1),
        cirq.phase_flip(0.1).on_each(q0, q1),
        circuit[4],
    )
    res2 = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)

    _assert_distributions_approx_eq(res1, res2)


def test_permutation_errors() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H.on_each(q0, q1, q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.H.on_each(q0, q1, q2),
    )

    model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(movement_phase_error=0.1)
    res = ess.simulation.leakage_sim.simulate_true_distribution(circuit.with_noise(model), 5)
    _assert_distributions_approx_eq(
        res,
        {
            "000": 0.9 * 0.9 * 0.9,
            "001": 0.9 * 0.9 * 0.1,
            "010": 0.9 * 0.1 * 0.9,
            "011": 0.9 * 0.1 * 0.1,
            "100": 0.1 * 0.9 * 0.9,
            "101": 0.1 * 0.9 * 0.1,
            "110": 0.1 * 0.1 * 0.9,
            "111": 0.1 * 0.1 * 0.1,
        },
    )
