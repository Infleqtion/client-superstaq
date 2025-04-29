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
from unittest.mock import patch

import cirq
import cirq.testing
import pandas as pd
import pytest

from supermarq.qcvv import CB, CBResults


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_bad_cb_init() -> None:
    with pytest.raises(
        RuntimeError, match="This cycle benchmarking is only valid for Clifford elements."
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
        CB(process, pauli_channels=["Y"])

    with pytest.raises(ValueError, match="Cycle benchmarking requires two factors"):
        qubits = cirq.LineQubit.range(2)
        process = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
        CB(process, pauli_channels=["Y"], process_order_factors=[1])


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
    channel = cirq.MutablePauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y}) #type: ignore
    aggregate_pauli = cirq.MutablePauliString(
        {qubits[0]: cirq.Z, qubits[1]: cirq.Z}, coefficient=1j
    ) #type: ignore
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
def cb_results() -> CBResults:
    qubits = cirq.LineQubit.range(2)
    process = cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.CX(qubits[0], qubits[1]),
            cirq.H(qubits[0]),
        ]
    )
    cb_experiment = CB(process, ["XY", "YZ"], 2, random_seed=0)
    data = pd.DataFrame(
        {
            "circuit_realization": [0, 1, 0, 1, 0, 1, 0, 1],
            "pauli_channel": ["YZ", "YZ", "YZ", "YZ", "XY", "XY", "XY", "XY"],
            "cycle_depth": [2, 2, 4, 4, 2, 2, 4, 4],
            "c_of_p": [
                (-cirq.Y(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1))).mutable_copy(),
                (-cirq.Z(cirq.LineQubit(1)) * cirq.Y(cirq.LineQubit(0))).mutable_copy(),
                (-cirq.Y(cirq.LineQubit(0)) * cirq.Z(cirq.LineQubit(1))).mutable_copy(),
                (-cirq.Z(cirq.LineQubit(1)) * cirq.Y(cirq.LineQubit(0))).mutable_copy(),
                ((1 + 0j) * cirq.Y(cirq.LineQubit(1)) * cirq.X(cirq.LineQubit(0))).mutable_copy(),
                (-cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1))).mutable_copy(),
                ((1 + 0j) * cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1))).mutable_copy(),
                (-cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1))).mutable_copy(),
            ],
            "circuit": ["process"] * 8,
            "00": [0.056, 0.068, 0.092, 0.114, 0.882, 0.062, 0.758, 0.118],
            "01": [0.85, 0.018, 0.774, 0.058, 0.04, 0.878, 0.114, 0.752],
            "10": [0.022, 0.862, 0.056, 0.758, 0.05, 0.02, 0.076, 0.054],
            "11": [0.072, 0.052, 0.078, 0.07, 0.028, 0.04, 0.052, 0.076],
        }
    )
    return CBResults("sim", cb_experiment, data=data)


def test_result_analyse(cb_results: CBResults) -> None:
    cb_results.analyze(plot_results=False, print_results=False)
