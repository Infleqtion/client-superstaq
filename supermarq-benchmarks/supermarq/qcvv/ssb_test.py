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
# pylint: disable=missing-function-docstring
# pylint: disable=missing-return-doc
# mypy: disable-error-code=method-assign

from __future__ import annotations

import itertools
import pathlib
import re
from unittest.mock import MagicMock

import cirq
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import SSB, SSBResults


def test_ssb_init() -> None:
    q0, q1 = cirq.LineQubit.range(2)

    experiment = SSB(num_circuits=10, cycle_depths=[2, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == [q0, q1]

    assert len(experiment._stabilizer_states) == 12
    assert len(experiment._init_rotations) == 12
    for rot in experiment._init_rotations:
        assert len(rot) == 5
    assert len(experiment._reconciliation_rotation) == 12
    for rot in experiment._reconciliation_rotation:
        assert len(rot) == 4


@pytest.fixture
def ssb_experiment() -> SSB:
    return SSB(num_circuits=10, cycle_depths=[2, 3, 5])


def test_random_parallel_qubit_rotation(ssb_experiment: SSB) -> None:

    ssb_experiment._rng = (rng := MagicMock())
    rng.choice.side_effect = [
        cirq.rx(np.pi / 2),
        cirq.ry(-np.pi / 2),
    ]
    moment = ssb_experiment._random_parallel_qubit_rotation()
    assert moment == cirq.Moment(
        cirq.rx(np.pi / 2)(cirq.LineQubit(0)), cirq.rx(np.pi / 2)(cirq.LineQubit(1))
    )

    moment = ssb_experiment._random_parallel_qubit_rotation()
    assert moment == cirq.Moment(
        cirq.ry(-np.pi / 2)(cirq.LineQubit(0)), cirq.ry(-np.pi / 2)(cirq.LineQubit(1))
    )


def test_sss_init_circuits(ssb_experiment: SSB) -> None:
    init_circuit = ssb_experiment._sss_init_circuits(4)
    # Index-4 has init_rotations [X, X, X, _Y, _X]
    q0, q1 = cirq.LineQubit.range(2)
    assert init_circuit == cirq.Circuit(
        cirq.X(q0),
        cirq.X(q1),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.CZ(q0, q1),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.Moment(cirq.ry(-np.pi / 2)(q0), cirq.ry(-np.pi / 2)(q1)),
        cirq.Moment(cirq.rx(-np.pi / 2)(q0), cirq.rx(-np.pi / 2)(q1)),
    )

def test_sss_reconciliation_circuit(ssb_experiment: SSB) -> None:
    init_circuit = ssb_experiment._sss_init_circuits(4)
    # Index-4 has recon_rotations [_X, X, X, X]

    recon_circuit = ssb_experiment._sss_reconciliation_circuit(init_circuit)
    q0, q1 = cirq.LineQubit.range(2)
    assert recon_circuit == cirq.Circuit(
        cirq.Moment(cirq.rx(-np.pi / 2)(q0), cirq.rx(-np.pi / 2)(q1)),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.CZ(q0, q1),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
        cirq.X(q0),
        cirq.X(q1),
    )


def test_build_ssb_circuit(ssb_experiment: SSB) -> None:

    ssb_experiment._rng = (rng := MagicMock())
    rng.choice.side_effect = [
        np.array([[cirq.X, cirq.Y], [cirq.Z, cirq.Y], [cirq.Y, cirq.Z]]),
        np.array([[cirq.X, cirq.Z], [cirq.X, cirq.X], [cirq.Y, cirq.Y]]),
    ]
    circuits = ssb_experiment._build_circuits(num_circuits=2, cycle_depths=[2])

    assert len(circuits) == 2

    qbs = ssb_experiment.qubits
    cirq.testing.assert_same_circuits(
        circuits[0].circuit,
        cirq.Circuit(
            [
                cirq.X(qbs[0]),
                cirq.Y(qbs[1]),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                cirq.Z(qbs[0]),
                cirq.Y(qbs[1]),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                cirq.Y(qbs[0]),
                cirq.Z(qbs[1]),
                cirq.measure(qbs),
            ]
        ),
    )
    assert circuits[0].data == {
        "circuit_depth": 5,
        "cycle_depth": 2,
        "two_qubit_gate": "CZ",
        "exact_00": 1.0,
        "exact_01": 0.0,
        "exact_10": 0.0,
        "exact_11": 0.0,
    }
    cirq.testing.assert_same_circuits(
        circuits[1].circuit,
        cirq.Circuit(
            [
                cirq.X(qbs[0]),
                cirq.Z(qbs[1]),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                cirq.X(qbs[0]),
                cirq.X(qbs[1]),
                cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
                cirq.Y(qbs[0]),
                cirq.Y(qbs[1]),
                cirq.measure(qbs),
            ]
        ),
    )
    assert circuits[1].data == {
        "circuit_depth": 5,
        "cycle_depth": 2,
        "two_qubit_gate": "CZ",
        "exact_00": 0.0,
        "exact_01": 0.0,
        "exact_10": 1.0,
        "exact_11": 0.0,
    }


def test_ssb_analyse_results(tmp_path: pathlib.Path, ssb_experiment: SSB) -> None:
    results = SSBResults(target="example", experiment=ssb_experiment)

    results.data = pd.DataFrame(
        [
            {
                "circuit_realization": 0,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
                "exact_00": 1.0,
                "exact_01": 0.0,
                "exact_10": 0.0,
                "exact_11": 0.0,
            },
            {
                "circuit_realization": 1,
                "cycle_depth": 1,
                "circuit_depth": 3,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
                "exact_00": 0.5,
                "exact_01": 0.5,
                "exact_10": 0.0,
                "exact_11": 0.0,
            },
            {
                "circuit_realization": 0,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 1.0,
                "10": 0.0,
                "11": 0.0,
                "exact_00": 0.0,
                "exact_01": 0.75,
                "exact_10": 0.25,
                "exact_11": 0.0,
            },
            {
                "circuit_realization": 1,
                "cycle_depth": 5,
                "circuit_depth": 11,
                "00": 0.0,
                "01": 0.0,
                "10": 1.0,
                "11": 0.0,
                "exact_00": 0.0,
                "exact_01": 0.5,
                "exact_10": 0.25,
                "exact_11": 0.25,
            },
            {
                "circuit_realization": 0,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.0,
                "01": 0.0,
                "10": 0.5,
                "11": 0.5,
                "exact_00": 0.2,
                "exact_01": 0.3,
                "exact_10": 0.25,
                "exact_11": 0.25,
            },
            {
                "circuit_realization": 1,
                "cycle_depth": 10,
                "circuit_depth": 21,
                "00": 0.0,
                "01": 0.0,
                "10": 0.5,
                "11": 0.5,
                "exact_00": 0.1,
                "exact_01": 0.1,
                "exact_10": 0.4,
                "exact_11": 0.4,
            },
        ]
    )

    plot_filename = tmp_path / "example.png"
    speckle_plot_filename = tmp_path / "example_speckle.png"

    results.analyze(plot_filename=plot_filename.as_posix())
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


def test_results_no_data() -> None:
    results = SSBResults(target="example", experiment=MagicMock(), data=None)
    with pytest.raises(RuntimeError, match="No data stored. Cannot perform analysis."):
        results._analyze()

    with pytest.raises(RuntimeError, match="No data stored. Cannot plot results."):
        results.plot_results()

    with pytest.raises(RuntimeError, match="No data stored. Cannot plot results."):
        results.plot_speckle()

    with pytest.raises(
        RuntimeError, match="No stored dataframe of circuit fidelities. Something has gone wrong."
    ):
        results.data = pd.DataFrame()
        results.plot_results()


def test_results_not_analyzed() -> None:
    results = SSBResults(target="example", experiment=MagicMock(), data=None)
    for attr in ["cycle_fidelity_estimate", "cycle_fidelity_estimate_std"]:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
        ):
            getattr(results, attr)


def test_dump_and_load(
    tmp_path_factory: pytest.TempPathFactory,
    ssb_experiment: SSB,
) -> None:
    filename = tmp_path_factory.mktemp("tempdir") / "file.json"
    ssb_experiment.to_file(filename)
    exp = SSB.from_file(filename)

    assert exp.samples == ssb_experiment.samples
    assert exp.num_qubits == ssb_experiment.num_qubits
    assert exp.num_circuits == ssb_experiment.num_circuits
    assert exp.cycle_depths == ssb_experiment.cycle_depths
    assert exp.single_qubit_gate_set == ssb_experiment.single_qubit_gate_set
    assert exp.two_qubit_gate == ssb_experiment.two_qubit_gate
