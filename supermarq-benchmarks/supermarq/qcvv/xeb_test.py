# Copyright 2025 Infleqtion
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
# mypy: disable-error-code=method-assign
from __future__ import annotations

import pathlib
import re
from unittest import mock

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import XEB, XEBResults


def test_xeb_init() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)

    experiment = XEB(num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)
    assert experiment.interleaved_layer == cirq.CZ(q0, q1)
    assert experiment.single_qubit_gate_set == [
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.0),
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25),
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.5),
    ]

    experiment = XEB(interleaved_layer=cirq.CX, num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)
    assert experiment.interleaved_layer == cirq.CX(q0, q1)
    assert experiment.single_qubit_gate_set == [
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.0),
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25),
        cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.5),
    ]

    experiment = XEB(single_qubit_gate_set=[cirq.X], num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)
    assert experiment.interleaved_layer == cirq.CZ(q0, q1)
    assert experiment.single_qubit_gate_set == [cirq.X]

    experiment = XEB(interleaved_layer=None, num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)
    assert not experiment.interleaved_layer

    interleaved_op = cirq.CCX(q2, q0, q1)
    experiment = XEB(interleaved_layer=interleaved_op, num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 3
    assert experiment.qubits == (q0, q1, q2)
    assert experiment.interleaved_layer == interleaved_op
    assert all(sample.circuit.all_qubits() == {q0, q1, q2} for sample in experiment.samples)

    interleaved_moment = cirq.Moment(cirq.Z(q2), cirq.H(q1))
    experiment = XEB(interleaved_layer=interleaved_moment, num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q1, q2)
    assert experiment.interleaved_layer == interleaved_moment
    assert all(sample.circuit.all_qubits() == {q1, q2} for sample in experiment.samples)

    interleaved_circuit = cirq.Circuit(cirq.CZ(q2, q0), cirq.CCZ(q0, q1, q2))
    experiment = XEB(interleaved_layer=interleaved_circuit, num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 3
    assert experiment.qubits == (q0, q1, q2)
    assert experiment.interleaved_layer == interleaved_circuit
    assert all(sample.circuit.all_qubits() == {q0, q1, q2} for sample in experiment.samples)


@pytest.fixture
def xeb_experiment() -> XEB:
    return XEB(
        single_qubit_gate_set=[cirq.X, cirq.Y, cirq.Z], num_circuits=10, cycle_depths=[1, 3, 5]
    )


def test_build_xeb_circuit(xeb_experiment: XEB) -> None:
    xeb_experiment._rng = (rng := mock.MagicMock())
    xeb_experiment.single_qubit_gate_set = [cirq.X, cirq.Y, cirq.Z]
    rng.integers.side_effect = [
        np.array([[0, 1]]),
        np.array([[2, 1], [2, 1]]),
        np.array([[0, 2]]),
        np.array([[1, 1], [2, 1]]),
    ]
    samples = xeb_experiment._build_circuits(num_circuits=2, cycle_depths=[2])

    assert len(samples) == 2

    qbs = xeb_experiment.qubits
    cirq.testing.assert_same_circuits(
        samples[0].circuit,
        cirq.Circuit(
            [
                cirq.X(qbs[0]),
                cirq.Y(qbs[1]),
                css.barrier(*qbs),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                css.barrier(*qbs),
                cirq.Z(qbs[0]),
                cirq.Z(qbs[1]),
                css.barrier(*qbs),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                css.barrier(*qbs),
                cirq.Y(qbs[0]),
                cirq.X(qbs[1]),
                cirq.measure(qbs),
            ]
        ),
    )
    assert samples[0].data == {
        "circuit_depth": 8,
        "cycle_depth": 2,
        "interleaved_layer": "CZ(q(0), q(1))",
    }
    cirq.testing.assert_same_circuits(
        samples[1].circuit,
        cirq.Circuit(
            [
                cirq.X(qbs[0]),
                cirq.Z(qbs[1]),
                css.barrier(*qbs),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                css.barrier(*qbs),
                cirq.Y(qbs[0]),
                cirq.X(qbs[1]),
                css.barrier(*qbs),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                css.barrier(*qbs),
                cirq.X(qbs[0]),
                cirq.Y(qbs[1]),
                cirq.measure(qbs),
            ]
        ),
    )
    assert samples[1].data == {
        "circuit_depth": 8,
        "cycle_depth": 2,
        "interleaved_layer": "CZ(q(0), q(1))",
    }


def test_single_qubit_gates_dont_repeat() -> None:
    xeb_experiment = XEB(num_circuits=10, cycle_depths=[10])
    for sample in xeb_experiment.samples:
        single_qubit_moments = [
            moment for moment in sample.circuit if all(cirq.num_qubits(g) == 1 for g in moment)
        ]
        for i, moment in enumerate(single_qubit_moments[:-1]):
            assert set(moment).isdisjoint(set(single_qubit_moments[i + 1]))

    # Exception when only one option
    xeb_experiment = XEB(num_circuits=2, cycle_depths=[10], single_qubit_gate_set=[cirq.H])
    for sample in xeb_experiment.samples:
        single_qubit_moments = [
            moment for moment in sample.circuit if all(cirq.num_qubits(g) == 1 for g in moment)
        ]
        assert set(single_qubit_moments) == {cirq.Moment(cirq.H.on_each(*xeb_experiment.qubits))}


def test_xeb_analyse_results(tmp_path: pathlib.Path, xeb_experiment: XEB) -> None:
    exp_data = pd.DataFrame(
        [
            {
                "uuid": xeb_experiment.samples[0].uuid,
                "circuit_realization": 0,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[1].uuid,
                "circuit_realization": 1,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[2].uuid,
                "circuit_realization": 0,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 1.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[3].uuid,
                "circuit_realization": 1,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 0.0,
                "10": 1.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[4].uuid,
                "circuit_realization": 0,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.0,
                "01": 0.0,
                "10": 0.5,
                "11": 0.5,
            },
            {
                "uuid": xeb_experiment.samples[5].uuid,
                "circuit_realization": 1,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.0,
                "01": 0.0,
                "10": 0.5,
                "11": 0.5,
            },
        ]
    )

    analytical_data = pd.DataFrame(
        [
            {
                "uuid": xeb_experiment.samples[0].uuid,
                "circuit_realization": 0,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[1].uuid,
                "circuit_realization": 1,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 0.5,
                "01": 0.5,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[2].uuid,
                "circuit_realization": 0,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 0.75,
                "10": 0.25,
                "11": 0.0,
            },
            {
                "uuid": xeb_experiment.samples[3].uuid,
                "circuit_realization": 1,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 0.5,
                "10": 0.25,
                "11": 0.25,
            },
            {
                "uuid": xeb_experiment.samples[4].uuid,
                "circuit_realization": 0,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.2,
                "01": 0.3,
                "10": 0.25,
                "11": 0.25,
            },
            {
                "uuid": xeb_experiment.samples[5].uuid,
                "circuit_realization": 1,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.1,
                "01": 0.1,
                "10": 0.4,
                "11": 0.4,
            },
        ]
    )

    plot_filename = tmp_path / "example.png"
    speckle_plot_filename = tmp_path / "example_speckle.png"

    results = XEBResults(target="example", experiment=xeb_experiment, data=exp_data)

    with mock.patch.object(results, "_analytical_data", return_value=analytical_data):
        results.analyze(plot_filename=plot_filename.as_posix())

    assert results.data is not None
    np.testing.assert_allclose(
        results.data["sum_p(x)p^(x)"].values, [1.0, 0.5, 0.75, 0.25, 0.25, 0.4]
    )
    np.testing.assert_allclose(
        results.data["sum_p(x)p(x)"].values, [1.0, 0.5, 0.625, 0.375, 0.255, 0.34]
    )

    # Calculated by hand
    assert results.cycle_fidelity_estimate == pytest.approx(1.0613025)
    assert results.cycle_fidelity_estimate_std == pytest.approx(0.0597633930)

    assert pathlib.Path(tmp_path / "example.png").exists()

    # Test the speckle plot
    results.plot_speckle(filename=speckle_plot_filename.as_posix())
    assert pathlib.Path(tmp_path / "example_speckle.png").exists()


@pytest.mark.parametrize(
    "layer",
    [
        None,
        cirq.Y,
        cirq.CCX,
        cirq.CX(cirq.q(4), cirq.q(0)),
        cirq.Moment(cirq.H(cirq.q(2)), cirq.X(cirq.q(0))),
        cirq.Circuit(cirq.CX(cirq.q(1), cirq.q(2)), cirq.CZ(cirq.q(2), cirq.q(0))),
    ],
)
def test_analytical_probabilities(
    layer: cirq.OP_TREE | None,
) -> None:
    xeb_experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=layer)

    def _exact_sim(circuit: cirq.Circuit) -> dict[int, float]:
        psi = circuit.final_state_vector(ignore_terminal_measurements=True, dtype=np.complex128)
        return dict(enumerate(np.abs(psi**2)))

    pd.testing.assert_frame_equal(
        xeb_experiment.run_with_callable(_exact_sim).data,
        xeb_experiment.run_with_simulator()._analytical_data(),
    )


def test_independent_qubit_groups() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)

    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=None)
    assert experiment.independent_qubit_groups() == [(q0, q1)]

    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=cirq.CZ)
    assert experiment.independent_qubit_groups() == [(q0, q1)]

    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=cirq.CCZ)
    assert experiment.independent_qubit_groups() == [(q0, q1, q2)]

    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=cirq.CZ(q1, q3))
    assert experiment.independent_qubit_groups() == [(q1, q3)]

    moment = cirq.Moment(cirq.X.on_each(q0, q2, q3))
    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=moment)
    assert experiment.independent_qubit_groups() == [(q0,), (q2,), (q3,)]

    moment = cirq.Moment(cirq.CZ(q1, q3), cirq.CZ(q0, q2))
    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=moment)
    assert experiment.independent_qubit_groups() == [(q0, q2), (q1, q3)]

    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.CZ(q1, q2), cirq.X(q3))
    experiment = XEB(num_circuits=3, cycle_depths=[1, 3], interleaved_layer=circuit)
    assert experiment.independent_qubit_groups() == [(q0, q1, q2), (q3,)]


def test_results_no_data() -> None:
    results = XEBResults(target="example", experiment=XEB(1, []), data=None)
    with pytest.raises(RuntimeError, match=r"No data stored. Cannot perform analysis."):
        results._analyze()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot plot results."):
        results.plot_results()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot plot results."):
        results.plot_speckle()

    with pytest.raises(
        RuntimeError, match=r"No stored dataframe of circuit fidelities. Something has gone wrong."
    ):
        results.data = pd.DataFrame()
        results.plot_results()


def test_results_not_analyzed() -> None:
    results = XEBResults(target="example", experiment=XEB(1, []), data=None)
    for attr in ["cycle_fidelity_estimate", "cycle_fidelity_estimate_std"]:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
        ):
            getattr(results, attr)


@pytest.mark.parametrize(
    "layer",
    [
        None,
        cirq.CX,
        cirq.CCX(cirq.q(2), cirq.q(0), cirq.q(1)),
        cirq.Moment(cirq.H(cirq.q(2)), cirq.X(cirq.q(0))),
        cirq.Circuit(cirq.CX(cirq.q(1), cirq.q(2)), cirq.CZ(cirq.q(2), cirq.q(0))),
    ],
)
def test_dump_and_load(
    tmp_path_factory: pytest.TempPathFactory,
    layer: cirq.OP_TREE | None,
) -> None:
    filename = tmp_path_factory.mktemp("tempdir") / "file.json"
    xeb_experiment = XEB(
        num_circuits=10,
        cycle_depths=[1, 3, 5],
        interleaved_layer=layer,
        single_qubit_gate_set=[cirq.X, cirq.Y, cirq.Z],
    )
    xeb_experiment.to_file(filename)
    exp = XEB.from_file(filename)

    assert exp.samples == xeb_experiment.samples
    assert exp.num_qubits == xeb_experiment.num_qubits
    assert exp.num_circuits == xeb_experiment.num_circuits
    assert exp.cycle_depths == xeb_experiment.cycle_depths
    assert exp.single_qubit_gate_set == xeb_experiment.single_qubit_gate_set
    assert exp.interleaved_layer == xeb_experiment.interleaved_layer
