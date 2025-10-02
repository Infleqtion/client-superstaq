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
import os
import pathlib
from unittest.mock import MagicMock, patch

import cirq
import cirq.testing
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import CB, CBResults


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_bad_cb_init() -> None:
    with pytest.raises(
        RuntimeError, match="Cycle Benchmarking is only valid for Clifford elements."
    ):
        qubit = cirq.LineQubit(0)
        process = cirq.Circuit([cirq.T(qubit)])
        CB(process, pauli_channels=1)

    with pytest.raises(
        RuntimeError, match="All Pauli channels must be over 1 qubits. XX is over 2 qubits."
    ):
        qubit = cirq.LineQubit(0)
        process = cirq.Circuit([cirq.X(qubit)])
        CB(process, pauli_channels=["XX"])

    with pytest.raises(
        RuntimeError, match="All Pauli channels must be over 2 qubits. Y is over 1 qubits."
    ):
        qubits = cirq.LineQubit.range(2)
        process = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
        CB(process, pauli_channels=["Y"])

    with pytest.raises(RuntimeError, match="The process circuit must not contain measurements."):
        qubits = cirq.LineQubit.range(2)
        process = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1]), cirq.M(qubits)])
        CB(process, pauli_channels=["YY"])

    with pytest.raises(ValueError, match="Cycle Benchmarking requires two factors"):
        qubits = cirq.LineQubit.range(2)
        process = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
        CB(process, pauli_channels=["YY"], process_order_factors=[1])

    with pytest.raises(TypeError, match="must be a list of Pauli strings or an integer"):
        qubits = cirq.LineQubit.range(2)
        process = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
        CB(process, pauli_channels="a")


def test_channels_cb_init() -> None:
    qubit = cirq.LineQubit(0)
    process = cirq.Circuit([cirq.X(qubit), cirq.H(qubit)])

    experiment = CB(process, pauli_channels=3)
    assert len(experiment.pauli_channels) == 3

    experiment = CB(process, pauli_channels=6)
    assert len(experiment.pauli_channels) == 4

    experiment = CB(process, pauli_channels=["X", "Z"])
    assert len(experiment.pauli_channels) == 2

    experiment = CB(process, pauli_channels=["X", "Z", "Z"])
    assert len(experiment.pauli_channels) == 2


def test_order_factors_cb_init() -> None:
    qubit = cirq.LineQubit(0)
    process = cirq.Circuit([cirq.X(qubit), cirq.H(qubit)])

    experiment = CB(process, 1)
    assert experiment.cycle_depths == [4, 8]

    experiment = CB(process, 1, process_order_factors=[2, 4])
    assert experiment.cycle_depths == [8, 16]


def test_undressed_cb_init() -> None:
    qubit = cirq.LineQubit(0)
    process = cirq.Circuit([cirq.X(qubit), cirq.H(qubit)])

    experiment = CB(process, 1)
    assert not experiment._undressed_process

    experiment = CB(process, 1, undressed_process=True)
    assert experiment._undressed_process


def test_num_samples_cb_init() -> None:
    qubit = cirq.LineQubit(0)
    process = cirq.Circuit([cirq.X(qubit), cirq.H(qubit)])

    experiment = CB(process, 2)
    assert len(experiment.samples) == 4

    experiment = CB(process, 2, num_circuits=2)
    assert len(experiment.samples) == 8

    experiment = CB(process, 2, num_circuits=2, undressed_process=True)
    assert len(experiment.samples) == 16


def test_find_process_order() -> None:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit([cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1])])
    compiled_circuit = CB._is_clifford(process)
    experiment = CB(process, pauli_channels=1)
    assert experiment._find_process_order(compiled_circuit) == 8

    with pytest.raises(RuntimeError, match="Could not find a circuit order less than 4"):
        experiment._find_process_order(compiled_circuit, max_depth=4)


@pytest.fixture
def cb_experiment() -> CB:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit([cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.H(qubits[0])])
    pauli_channels = ["XY"]
    with patch("supermarq.qcvv.cb.np.random.default_rng") as mock_rng:
        mock_rng.return_value.choice.side_effect = (
            [[p1, p2] for p1, p2 in itertools.product("XY", "YZ")]
            + [[p1, p2] for p1, p2 in itertools.product("ZX", "XY")]
            + [["X", "Y"], ["X", "Z"], ["Y", "Y"]] * 2
        )
        experiment = CB(process, pauli_channels=pauli_channels)
    return experiment


def test_generate_random_pauli() -> None:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit([cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.H(qubits[0])])
    experiment = CB(process, 1)
    experiment._rng = (rng := MagicMock())
    rng.choice.side_effect = [["X", "X"], ["X", "X"], ["X", "Y"]]
    pauli_strings = experiment._generate_random_pauli_strings(2)
    assert pauli_strings == ["XX", "XY"]


def test_state_prep_circuit(cb_experiment: CB) -> None:
    circuit, pauli_string = cb_experiment._state_prep_circuit("XY")
    qubits = cb_experiment.qubits
    cirq.testing.assert_same_circuits(
        circuit, cirq.Circuit([cirq.Y(qubits[0]) ** 0.5, cirq.X(qubits[1]) ** (-0.5)])
    )

    assert pauli_string == cirq.MutablePauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})


def test_cb_bulk_circuit(cb_experiment: CB) -> None:
    circuit, pauli_string = cb_experiment._cb_bulk_circuit(cb_experiment.cycle_depths[0])
    qubits = cb_experiment.qubits
    cirq.testing.assert_same_circuits(
        circuit,
        cirq.Circuit(
            [
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Z(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.Y(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
            ]
        ),
    )

    assert pauli_string == cirq.MutablePauliString(
        {qubits[0]: cirq.Z, qubits[1]: cirq.Z}, coefficient=1j
    )


def test_inversion_circuit(cb_experiment: CB) -> None:
    qubits = cb_experiment.qubits
    channel: cirq.MutablePauliString[cirq.Qid]
    channel = cirq.MutablePauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})
    aggregate_pauli: cirq.MutablePauliString[cirq.Qid]
    aggregate_pauli = cirq.MutablePauliString(
        {qubits[0]: cirq.Z, qubits[1]: cirq.Z}, coefficient=1j
    )
    circuit, pauli_string = cb_experiment._inversion_circuit(channel, aggregate_pauli)

    cirq.testing.assert_same_circuits(
        circuit, cirq.Circuit([cirq.Y(qubits[0]) ** (-0.5), cirq.X(qubits[1]) ** 0.5])
    )

    assert pauli_string == cirq.MutablePauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})


def test_generate_full_circuit(cb_experiment: CB) -> None:
    circuit, pauli_string = cb_experiment._generate_full_cb_circuit(
        "XY", cb_experiment.cycle_depths[0]
    )
    qubits = cb_experiment.qubits

    cirq.testing.assert_same_circuits(
        circuit,
        cirq.Circuit(
            [
                cirq.Moment(
                    cirq.Y(qubits[0]) ** 0.5,
                    cirq.X(qubits[1]) ** (-0.5),
                ),
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Z(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.Y(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
                cirq.Moment(cirq.Y(qubits[0]) ** (-0.5), cirq.X(qubits[1]) ** 0.5),
                cirq.M(qubits),
            ]
        ),
    )

    assert pauli_string == cirq.MutablePauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})


def test_from_json_dict() -> None:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit([cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.H(qubits[0])])

    experiment = CB._from_json_dict_(
        process_circuit=cirq.to_json(process),
        pauli_channels=["XX", "YY"],
        num_circuits=2,
        process_order_factors=[2, 4],
        undressed_process=False,
    )

    assert {m for m in experiment.pauli_channels} == {"XX", "YY"}
    assert experiment.cycle_depths == [4, 8]
    assert not experiment._undressed_process
    assert len(experiment.samples) == 8


def test_json_dict() -> None:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit([cirq.H(qubits[0]), cirq.CX(qubits[0], qubits[1]), cirq.H(qubits[0])])
    expected_circuit = cirq.Circuit()
    for op in process.all_operations():
        expected_circuit += op.with_tags("no_compile")
    experiment = CB(process, ["XX", "YY"], 1, [2, 4], False)

    json = experiment._json_dict_()

    assert json["process_circuit"] == cirq.to_json(expected_circuit)
    assert set(json["pauli_channels"]) == {"XX", "YY"}
    assert json["num_circuits"] == 1
    assert json["process_order_factors"] == [2, 4]
    assert not json["undressed_process"]


def test_samples_cb_experiment(cb_experiment: CB) -> None:
    samples = cb_experiment.samples
    qubits = cb_experiment.qubits

    assert len(samples) == 2

    cirq.testing.assert_same_circuits(
        samples[0].circuit,
        cirq.Circuit(
            [
                cirq.Moment(
                    cirq.Y(qubits[0]) ** 0.5,
                    cirq.X(qubits[1]) ** (-0.5),
                ),
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.X(qubits[0]),
                    cirq.Z(qubits[1]),
                ),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.Moment(
                    cirq.Y(qubits[0]),
                    cirq.Y(qubits[1]),
                ),
                cirq.Moment(cirq.Y(qubits[0]) ** (-0.5), cirq.X(qubits[1]) ** 0.5),
                cirq.M(qubits),
            ]
        ),
    )

    assert samples[0].data["pauli_channel"] == "XY"
    assert samples[1].data["pauli_channel"] == "XY"
    assert samples[0].data["c_of_p"] == cirq.MutablePauliString(
        {qubits[0]: cirq.X, qubits[1]: cirq.Y}
    )
    assert samples[1].data["c_of_p"] == cirq.MutablePauliString(
        {qubits[0]: cirq.X, qubits[1]: cirq.Y}, coefficient=-1
    )


@pytest.fixture
def dressed_cb_results() -> CBResults:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.CX(qubits[0], qubits[1]),
            cirq.H(qubits[0]),
        ]
    )
    dressed_cb_experiment = CB(process, ["XY", "YZ"], 2, random_seed=0)
    dressed_data = pd.DataFrame(
        {
            "circuit_realization": [0, 1] * 4,
            "pauli_channel": ["YZ"] * 4 + ["XY"] * 4,
            "cycle_depth": [2, 2, 4, 4] * 2,
            "c_of_p": [
                -cirq.Y(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1)),
                -cirq.Z(cirq.LineQubit(1)) * cirq.Y(cirq.LineQubit(0)),
                -cirq.Y(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1)),
                -cirq.Z(cirq.LineQubit(1)) * cirq.Y(cirq.LineQubit(0)),
                (1 + 0j) * cirq.Y(cirq.LineQubit(1)) * cirq.X(cirq.LineQubit(0)),
                -cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                (1 + 0j) * cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                -cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
            ],
            "circuit": ["process"] * 8,
            "00": [0.05, 0.075, 0.1, 0.1, 0.9, 0.05, 0.75, 0.1],
            "01": [0.85, 0.025, 0.75, 0.05, 0.025, 0.9, 0.1, 0.75],
            "10": [0.025, 0.85, 0.05, 0.75, 0.05, 0.025, 0.075, 0.075],
            "11": [0.075, 0.05, 0.1, 0.1, 0.025, 0.025, 0.075, 0.075],
        }
    )
    return CBResults("sim", dressed_cb_experiment, data=dressed_data)


@pytest.fixture
def undressed_cb_results() -> CBResults:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.CX(qubits[0], qubits[1]),
            cirq.H(qubits[0]),
        ]
    )
    undressed_cb_experiment = CB(process, ["XY"], 2, undressed_process=True, random_seed=0)
    undressed_data = pd.DataFrame(
        {
            "circuit_realization": [0, 0, 1, 1] * 2,
            "pauli_channel": ["XY"] * 8,
            "cycle_depth": [2] * 4 + [4] * 4,
            "c_of_p": [
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                -cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                -cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)),
            ],
            "circuit": ["process", "identity"] * 4,
            "00": [0.9, 0.05, 0.025, 0.9, 0.15, 0.8, 0.8, 0.8],
            "01": [0.05, 0.025, 0.025, 0.05, 0.025, 0.025, 0.025, 0.15],
            "10": [0.025, 0.9, 0.05, 0.025, 0.8, 0.025, 0.15, 0.025],
            "11": [0.024, 0.025, 0.9, 0.025, 0.025, 0.15, 0.025, 0.025],
        }
    )
    return CBResults("sim", undressed_cb_experiment, data=undressed_data)


@pytest.mark.parametrize("results_name", ["dressed_cb_results", "undressed_cb_results"])
def test_results_not_analysed(results_name: str, request: pytest.FixtureRequest) -> None:
    results = request.getfixturevalue(results_name)
    for attr in ["channel_fidelities", "process_fidelity", "process_fidelity_std"]:
        with pytest.raises(RuntimeError, match="Value has not yet been estimated"):
            getattr(results, attr)


def test_results_not_analysed_undressed(undressed_cb_results: CBResults) -> None:
    for attr in ["undressed_process_fidelity", "undressed_process_fidelity_std"]:
        with pytest.raises(RuntimeError, match="Value has not yet been estimated"):
            getattr(undressed_cb_results, attr)


def test_results_undressed_with_dressed(dressed_cb_results: CBResults) -> None:
    for attr in ["undressed_process_fidelity", "undressed_process_fidelity_std"]:
        with pytest.raises(
            RuntimeError,
            match="Undressed process fidelity is not available for this experiment.",
        ):
            getattr(dressed_cb_results, attr)


def test_results_analyse(dressed_cb_results: CBResults) -> None:
    dressed_cb_results.analyze(plot_results=False, print_results=False)

    np.testing.assert_allclose(
        dressed_cb_results._channel_expectations["expectation_mean"].values,
        [0.85, 0.65, 0.75, 0.60],
    )
    np.testing.assert_allclose(
        dressed_cb_results._channel_expectations["expectation_delta"].values,
        [0, 0, 0, 0],
        atol=1e-2,
    )
    np.testing.assert_allclose(
        dressed_cb_results.channel_fidelities["fidelity"].values,
        [np.sqrt(0.65 / 0.85), np.sqrt(0.6 / 0.75)],
        atol=1e-6,
    )


@pytest.mark.parametrize("results_name", ["dressed_cb_results", "undressed_cb_results"])
def test_plot_results(results_name: str, request: pytest.FixtureRequest) -> None:
    results = request.getfixturevalue(results_name)
    results.analyze(plot_results=True, print_results=False)


@pytest.mark.parametrize("results_name", ["dressed_cb_results", "undressed_cb_results"])
def test_print_results(results_name: str, request: pytest.FixtureRequest) -> None:
    results = request.getfixturevalue(results_name)
    results.analyze(plot_results=False, print_results=True)


@pytest.mark.parametrize("results_name", ["dressed_cb_results", "undressed_cb_results"])
def test_save_plot_results(
    results_name: str, request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> None:
    filename = tmp_path / "test_plot.png"
    results = request.getfixturevalue(results_name)
    results.analyze(plot_results=True, print_results=False, plot_filename=filename.as_posix())
    assert pathlib.Path(filename).exists()
