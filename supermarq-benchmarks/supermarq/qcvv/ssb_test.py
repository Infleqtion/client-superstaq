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

import pathlib
import re
from unittest.mock import MagicMock

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import SSB, SSBResults


def test_ssb_init() -> None:
    q0, q1 = cirq.LineQubit.range(2)

    experiment = SSB(num_circuits=10, cycle_depths=[2, 3, 5])
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)

    assert len(experiment._stabilizer_states) == 12
    assert len(experiment._init_rotations) == 12
    for rot in experiment._init_rotations:
        assert len(rot) == 5
    assert len(experiment._reconciliation_rotation) == 12
    for rot in experiment._reconciliation_rotation:
        assert len(rot) == 4

    with pytest.raises(ValueError, match="Cannot perform SSB with a cycle depth of 1."):
        SSB(num_circuits=10, cycle_depths=[1, 2, 3, 5])


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
    assert moment == cirq.rx(np.pi / 2)

    moment = ssb_experiment._random_parallel_qubit_rotation()
    assert moment == cirq.ry(-np.pi / 2)


def test_sss_init_circuits(ssb_experiment: SSB) -> None:
    init_circuit = ssb_experiment._sss_init_circuit(4)
    # Index-4 has init_rotations [X, X, X, _Y, _X]
    q0, q1 = cirq.LineQubit.range(2)
    assert init_circuit == cirq.Circuit(
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
        css.ParallelRGate(np.pi / 2, 0, 2)(q0, q1),
        cirq.CZ(q0, q1),
        css.ParallelRGate(np.pi / 2, 0, 2)(q0, q1),
        css.ParallelRGate(np.pi / 2, -np.pi / 2, 2)(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
    )


def test_sss_reconciliation_circuit(ssb_experiment: SSB) -> None:
    init_circuit = ssb_experiment._sss_init_circuit(4)
    # Index-4 has recon_rotations [_X, X, X, X]

    recon_circuit = ssb_experiment._sss_reconciliation_circuit(init_circuit)
    q0, q1 = cirq.LineQubit.range(2)
    assert recon_circuit == cirq.Circuit(
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
        cirq.CZ(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
        css.ParallelRGate(np.pi / 2, np.pi, 2)(q0, q1),
    )


def test_build_ssb_circuit(ssb_experiment: SSB) -> None:
    ssb_experiment._rng = (rng := MagicMock())
    rng.integers.return_value = 4
    rng.choice.return_value = cirq.rx(np.pi / 2)
    circuits = ssb_experiment._build_circuits(num_circuits=1, cycle_depths=[2])

    assert len(circuits) == 1

    q0, q1 = ssb_experiment.qubits
    cirq.testing.assert_same_circuits(
        circuits[0].circuit,
        cirq.Circuit(
            # Init circuit
            cirq.X(q0),
            cirq.X(q1),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.CZ(q0, q1).with_tags("no_compile"),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.ry(-np.pi / 2)(q0), cirq.ry(-np.pi / 2)(q1)),
            cirq.Moment(cirq.rx(-np.pi / 2)(q0), cirq.rx(-np.pi / 2)(q1)),
            # Intermediate ops
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            # Reconcilliation
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.CZ(q0, q1).with_tags("no_compile"),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.Moment(cirq.rx(np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)),
            cirq.X(q0),
            cirq.X(q1),
            cirq.Moment(
                cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1)),
            ),
        ),
    )
    assert circuits[0].data == {
        "initial_sss_index": 4,
        "num_cz_gates": 2,
    }


def test_ssb_analyse_results(tmp_path: pathlib.Path, ssb_experiment: SSB) -> None:
    results = SSBResults(target="example", experiment=ssb_experiment)

    results.data = pd.DataFrame(
        [
            {
                "initial_sss_index": 4,
                "num_cz_gates": 2,
                "00": 1.0,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "initial_sss_index": 4,
                "num_cz_gates": 3,
                "00": 0.9,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "initial_sss_index": 4,
                "num_cz_gates": 4,
                "00": 0.8,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "initial_sss_index": 4,
                "num_cz_gates": 5,
                "00": 0.7,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
            {
                "initial_sss_index": 4,
                "num_cz_gates": 6,
                "00": 0.6,
                "01": 0.0,
                "10": 0.0,
                "11": 0.0,
            },
        ]
    )

    plot_filename = tmp_path / "example.png"

    results.analyze(plot_filename=plot_filename.as_posix())

    # Calculated by hand
    assert results.cz_fidelity_estimate == pytest.approx(0.81358891968418)
    assert results.cz_fidelity_estimate_std == pytest.approx(0.14459611416597062)

    assert pathlib.Path(tmp_path / "example.png").exists()


def test_results_no_data() -> None:
    results = SSBResults(target="example", experiment=MagicMock(), data=None)
    with pytest.raises(RuntimeError, match="No data stored. Cannot perform analysis."):
        results._analyze()

    with pytest.raises(RuntimeError, match="No data stored. Cannot plot results."):
        results.plot_results()


def test_results_not_analyzed() -> None:
    results = SSBResults(target="example", experiment=MagicMock(), data=None)
    for attr in ["cz_fidelity_estimate", "cz_fidelity_estimate_std"]:
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
    assert exp.samples == ssb_experiment.samples
