# Copyright (c) 2026 Infleqtion. All Rights Reserved.
#
# This file and its contents are the proprietary property of Infleqtion and may not be
# disclosed, copied, distributed, or used without prior written authorization from Infleqtion.
# No open-source license applies to this code. Unauthorized use is strictly prohibited.
from __future__ import annotations

from collections.abc import Mapping

import cirq
import cirq_superstaq as css
import numpy as np
import pytest
import sympy

import experiments_superstaq as ess


def test_examples() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0, 3).on(q0, q1, q2),
        cirq.CZ(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.rz(np.pi / 2).on(q1),
        css.ParallelRGate(-np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.measure(q0, q1, q2, key="c"),
    )

    noise_params = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS
    noise_model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(**noise_params.model_dump())
    noisy_circuit = circuit.with_noise(noise_model)

    # Exact simulation using density matrix
    _ = ess.simulation.leakage_sim.simulate_true_distribution(noisy_circuit, dimension=5)

    # Estimate using clifford + leakage simulator
    _ = ess.simulation.leakage_sim.estimate_distribution(
        noisy_circuit, repetitions=10, oversample=10, progressbar=False
    )
    _ = ess.simulation.leakage_sim.estimate_distribution(
        noisy_circuit, repetitions=10, oversample=10, progressbar=True
    )

    # Run the simulator as a sampler
    _ = ess.simulation.leakage_sim.sample_circuit(noisy_circuit, repetitions=10, progressbar=False)
    _ = ess.simulation.leakage_sim.sample_circuit(noisy_circuit, repetitions=10, progressbar=True)


def test_leakage_state() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    state = ess.simulation.leakage_sim.LeakageState([q0, q1, q2])
    copied_state = state.copy()
    assert copied_state.state == state.state
    assert copied_state._sim_states == state._sim_states
    assert state._sim_states == {
        i: ess.simulation.leakage_sim.ClassicalDistribution(0) for i in range(3)
    }

    for op in [
        cirq.X(q0),
        cirq.H(q1),
        cirq.CX(q1, q2),
        css.barrier(q0, q1, q2),
    ]:
        cirq.act_on(op, state)

    assert state._sim_states == {0: ess.simulation.leakage_sim.ClassicalDistribution(1)}
    assert state._in_classical_state(q0)
    assert not state._in_classical_state(q1)
    assert not state._in_classical_state(q2)

    cirq.act_on(cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2), state)
    assert state._in_classical_state(q1)
    assert not state._in_classical_state(q2)
    assert not state._in_classical_state(q0)

    copied_state = state.copy()
    state._act_on_fallback_(css.barrier(q0, q1, q2), [q0, q1, q2])
    assert copied_state._state == state._state
    assert copied_state._sim_states == state._sim_states

    x = sympy.Symbol("x")
    assert state._act_on_fallback_(cirq.X**x, [q1]) is NotImplemented
    assert state._act_on_fallback_(cirq.ParallelGate(cirq.X**x, 2), [q0, q1]) is NotImplemented

    q0, q1, q2 = cirq.LineQubit.range(3)
    state = ess.simulation.leakage_sim.LeakageState([q0, q1, q2])

    transition_matrix = [
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.1],
    ]
    cirq.act_on(cirq.X(q1), state)
    cirq.act_on(cirq.H(q2), state)

    cirq.act_on(ess.simulation.JumpChannel(transition_matrix).on(q0), state)
    np.testing.assert_allclose(state._sim_states[0]._state, [0.0, 0.0, 1.0, 0.0, 0.0])

    cirq.act_on(ess.simulation.JumpChannel(transition_matrix).on(q0), state)
    np.testing.assert_allclose(state._sim_states[0]._state, [0.0, 0.0, 0.9, 0.0, 0.1])

    cirq.act_on(ess.simulation.JumpChannel(transition_matrix).on(q1), state)
    np.testing.assert_allclose(state._sim_states[1]._state, [0.2, 0.3, 0.0, 0.5, 0.0])

    cirq.act_on(ess.simulation.JumpChannel(transition_matrix).on(q1), state)
    np.testing.assert_allclose(state._sim_states[1]._state, [0.06, 0.09, 0.2, 0.65, 0.0])

    cirq.act_on(ess.simulation.JumpChannel(transition_matrix).on(q2), state)
    assert np.allclose(state._sim_states[2]._state, [0.0, 0.0, 1.0, 0.0, 0.0]) or np.allclose(
        state._sim_states[2]._state, [0.2, 0.3, 0.0, 0.5, 0.0]
    )


def test_decompose_parallel_ops() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.ParallelGate(cirq.H, 2).on(q0, q2),
        css.ParallelRGate(np.pi / 2, np.pi, 3).on(q0, q1, q2),
        cirq.IdentityGate(2).on(q1, q3),
        css.barrier(q2, q3),
        cirq.I(q0),
        cirq.measure(q2),
    )
    expected = cirq.Circuit(
        cirq.Moment(cirq.H.on_each(q0, q2)),
        cirq.Moment(css.RGate(np.pi / 2, np.pi).on_each(q0, q1, q2)),
        cirq.Moment(cirq.measure(q2)),
    )
    cirq.testing.assert_same_circuits(
        ess.simulation.leakage_sim._decompose_parallel_ops(circuit),
        expected,
    )


def test_optimize_for_simulation() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        css.ParallelRGate(1, 2, 3).on(q0, q1, q2),
        cirq.X(q3),
        css.barrier(q0, q1, q2, q3),
        cirq.measure(q1),
    )
    cirq.testing.assert_same_circuits(
        ess.simulation.leakage_sim._optimize_for_simulation(circuit),
        cirq.Circuit(
            css.RGate(1, 2).on(q1),
            cirq.measure(q1),
        ),
    )

    circuit += cirq.measure(q2, q3)
    cirq.testing.assert_same_circuits(
        ess.simulation.leakage_sim._optimize_for_simulation(circuit),
        cirq.Circuit(
            css.RGate(1, 2).on(q1),
            css.RGate(1, 2).on(q2),
            cirq.X(q3),
            cirq.measure(q1),
            cirq.measure(q2, q3),
        ),
    )

    circuit = cirq.Circuit(
        css.ParallelRGate(1, 2, 3).on(q0, q1, q2),
        cirq.X(q3),
        cirq.SWAP(q2, q3),
        cirq.SWAP(q2, q1),
        cirq.measure(q1),
    )
    cirq.testing.assert_same_circuits(
        ess.simulation.leakage_sim._optimize_for_simulation(circuit),
        cirq.Circuit(
            cirq.X(q1),
            cirq.measure(q1),
        ),
    )

    circuit = cirq.Circuit(
        css.ParallelRGate(1, 2, 3).on(q0, q1, q2),
        cirq.X(q3),
        cirq.QubitPermutationGate([2, 0, 1]).on(q0, q1, q3),
        cirq.measure(q1, q0),
    )
    cirq.testing.assert_same_circuits(
        ess.simulation.leakage_sim._optimize_for_simulation(circuit),
        cirq.Circuit(
            css.RGate(1, 2).on(q0),
            cirq.X(q1),
            cirq.measure(q1, q0),
        ),
    )


def test_leakage_state_twirl_approx() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    state = ess.simulation.leakage_sim.LeakageState([q0, q1, q2], seed=1234)

    cirq.act_on(cirq.rx(1e-4), state, [q0])

    cirq.act_on(cirq.H, state, [q1])
    cirq.act_on(cirq.rz(1e-4), state, [q1])
    cirq.act_on(cirq.H, state, [q1])

    cirq.act_on(cirq.rx(np.pi - 1e-4), state, [q2])

    # Chance of small rotation causing a bit flip is ~1.5e-9
    samples = state.sample([q0, q1, q2], 100)
    np.testing.assert_array_equal(samples, 100 * [[0, 0, 1]])


def _assert_distributions_approx_eq(
    dist0: Mapping[str, float],
    dist1: Mapping[str, float],
    atol: float = 1e-8,
) -> None:
    assert all(
        np.isclose(dist0.get(key, 0), dist1.get(key, 0), atol=atol)
        for key in dist0.keys() | dist1.keys()
    )


@pytest.mark.parametrize("ch_form", [False, True])
@pytest.mark.parametrize("max_workers", [0, None])
def test_sample_circuit(ch_form: bool, max_workers: int | None) -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0, 3).on(q0, q1, q2),
        cirq.CZ(q0, q1),
        css.barrier(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.rz(np.pi / 2).on(q2),
        css.ParallelRGate(-np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.measure(q1, q2, key="c"),
    )

    res = ess.simulation.leakage_sim.sample_circuit(
        circuit, repetitions=10, ch_form=ch_form, max_workers=max_workers
    )
    assert res.measurements.keys() == {"c"}
    dist = res.histogram(key="c", fold_func=lambda m: "".join(map(str, m)))
    assert dist.keys() <= {"00", "11"}
    assert sum(dist.values()) == 10

    res = ess.simulation.leakage_sim.sample_circuit(
        circuit, repetitions=10, oversample=5, ch_form=ch_form, max_workers=max_workers
    )
    assert res.measurements.keys() == {"c"}
    dist = res.histogram(key="c", fold_func=lambda m: "".join(map(str, m)))
    assert dist.keys() <= {"00", "11"}
    assert sum(dist.values()) == 50

    noise_params = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS
    noise_model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(**noise_params.model_dump())
    noisy_circuit = circuit.with_noise(noise_model)

    res = ess.simulation.leakage_sim.sample_circuit(
        noisy_circuit, repetitions=100, oversample=5, ch_form=ch_form, max_workers=max_workers
    )
    assert res.measurements.keys() == {"c"}
    dist = res.histogram(key="c", fold_func=lambda m: "".join(map(str, m)))
    assert dist.keys() <= {"00", "01", "02", "10", "11", "12", "20", "21", "22"}
    assert sum(dist.values()) == 500

    circuit = cirq.Circuit(
        cirq.X.on_each(q1, q2),
        cirq.measure(q0, q1, key="c"),
        cirq.measure(q2, key="d"),
        cirq.X.on_each(q0, q1),
        cirq.measure(q0, q1, key="c"),
        cirq.measure(q2, key="d"),
    )
    res = ess.simulation.leakage_sim.sample_circuit(
        circuit, repetitions=10, ch_form=ch_form, max_workers=max_workers
    )
    np.testing.assert_array_equal(res.records["c"], 10 * [[[0, 1], [1, 0]]])
    np.testing.assert_array_equal(res.records["d"], 10 * [[[1], [1]]])


@pytest.mark.parametrize("ch_form", [False, True])
@pytest.mark.parametrize("oversample", [0, 10])
@pytest.mark.parametrize("max_workers", [0, None])
def test_estimate_distribution(ch_form: bool, oversample: int, max_workers: int | None) -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0, 3).on(q0, q1, q2),
        cirq.CZ(q0, q1),
        css.barrier(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.rz(np.pi / 2).on(q1),
        css.ParallelRGate(-np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
    )

    dist = ess.simulation.leakage_sim.estimate_distribution(
        circuit, repetitions=10, oversample=oversample, ch_form=ch_form, max_workers=max_workers
    )
    assert dist.keys() <= {"000", "011", "100", "111"}
    assert np.isclose(sum(dist.values()), 1.0)

    circuit += cirq.measure(q1, q2, key="c")
    dist = ess.simulation.leakage_sim.estimate_distribution(
        circuit, repetitions=10, oversample=oversample, ch_form=ch_form, max_workers=max_workers
    )
    assert dist.keys() <= {"00", "11"}
    assert np.isclose(sum(dist.values()), 1.0)

    noise_params = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS
    noise_model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(**noise_params.model_dump())
    noisy_circuit = circuit.with_noise(noise_model)
    dist = ess.simulation.leakage_sim.estimate_distribution(
        noisy_circuit,
        repetitions=100,
        oversample=oversample,
        ch_form=ch_form,
        max_workers=max_workers,
    )
    assert dist.keys() <= {"00", "01", "02", "10", "11", "12", "20", "21", "22"}
    assert np.isclose(sum(dist.values()), 1.0)


def test_simulate_true_distribution() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        css.ParallelRGate(np.pi / 2, 0, 3).on(q0, q1, q2),
        cirq.CZ(q0, q1),
        css.barrier(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
        cirq.QubitPermutationGate([1, 2, 0]).on(q0, q1, q2),
        cirq.rz(np.pi / 2).on(q1),
        css.ParallelRGate(-np.pi / 2, np.pi / 2, 3).on(q0, q1, q2),
    )

    dist = ess.simulation.leakage_sim.simulate_true_distribution(circuit, dimension=2)
    _assert_distributions_approx_eq(dist, {"000": 0.25, "011": 0.25, "100": 0.25, "111": 0.25})

    circuit += cirq.measure(q1, q2, key="c")
    dist = ess.simulation.leakage_sim.simulate_true_distribution(circuit, dimension=4)
    _assert_distributions_approx_eq(dist, {"00": 0.5, "11": 0.5})

    noise_params = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS
    noise_model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(**noise_params.model_dump())
    noisy_circuit = circuit.with_noise(noise_model)
    dist = ess.simulation.leakage_sim.simulate_true_distribution(noisy_circuit, dimension=5)
    assert dist.keys() == {"00", "01", "02", "10", "11", "12", "20", "21", "22"}


def test_bit_ordering() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3, 9, 2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.H(q1),
        css.ParallelRGate(-np.pi, np.pi / 2, 3).on(q0, q1, q2),
    )

    dist = ess.simulation.leakage_sim.simulate_true_distribution(circuit, dimension=5)
    _assert_distributions_approx_eq(dist, {"001": 0.5, "011": 0.5})

    measured_circuit = circuit + cirq.measure(q0, q2, key="a")
    result = ess.simulation.leakage_sim.sample_circuit(measured_circuit, 1)
    sdist = result.histogram(key="a", fold_func=lambda bs: "".join(map(str, bs)))
    edist = ess.simulation.leakage_sim.estimate_distribution(measured_circuit)
    tdist = ess.simulation.leakage_sim.simulate_true_distribution(measured_circuit, 3)
    assert tdist == edist == sdist == {"01": 1.0}

    measured_circuit = circuit + cirq.measure(q2, q0, key="a")
    result = ess.simulation.leakage_sim.sample_circuit(measured_circuit, 1)
    sdist = result.histogram(key="a", fold_func=lambda bs: "".join(map(str, bs)))
    edist = ess.simulation.leakage_sim.estimate_distribution(measured_circuit)
    tdist = ess.simulation.leakage_sim.simulate_true_distribution(measured_circuit, 3)
    assert tdist == edist == sdist == {"10": 1.0}

    measured_circuit = circuit + cirq.measure(q0, key="a") + cirq.measure(q2, key="b")
    result = ess.simulation.leakage_sim.sample_circuit(measured_circuit, 1)
    sdist = result.multi_measurement_histogram(
        keys=("a", "b"), fold_func=lambda bs: "".join(map(str, np.concatenate(bs)))
    )
    edist = ess.simulation.leakage_sim.estimate_distribution(measured_circuit)
    tdist = ess.simulation.leakage_sim.simulate_true_distribution(measured_circuit, 3)
    assert tdist == edist == sdist == {"01": 1.0}

    measured_circuit = circuit + cirq.measure(q0, key="b") + cirq.measure(q2, key="a")
    result = ess.simulation.leakage_sim.sample_circuit(measured_circuit, 1)
    sdist = result.multi_measurement_histogram(
        keys=("a", "b"), fold_func=lambda bs: "".join(map(str, np.concatenate(bs)))
    )
    edist = ess.simulation.leakage_sim.estimate_distribution(measured_circuit)
    tdist = ess.simulation.leakage_sim.simulate_true_distribution(measured_circuit, 3)
    assert tdist == edist == sdist == {"10": 1.0}
