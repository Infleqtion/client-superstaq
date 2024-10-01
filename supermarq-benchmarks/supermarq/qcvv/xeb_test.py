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
from unittest.mock import MagicMock, patch

import cirq
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import XEB, XEBSample


def test_xeb_init() -> None:
    experiment = XEB()
    assert experiment.num_qubits == 2
    assert experiment.two_qubit_gate == cirq.CZ
    assert experiment.single_qubit_gate_set == [
        cirq.PhasedXZGate(
            z_exponent=z,
            x_exponent=0.5,
            axis_phase_exponent=a,
        )
        for a, z in itertools.product(np.linspace(start=0, stop=7 / 4, num=8), repeat=2)
    ]

    with pytest.raises(
        RuntimeError, match="No samples to retrieve. The experiment has not been run."
    ):
        experiment.samples  # pylint: disable=W0104

    with pytest.raises(RuntimeError, match="No data to retrieve. The experiment has not been run."):
        experiment.circuit_fidelities  # pylint: disable=W0104

    experiment = XEB(two_qubit_gate=cirq.CX)
    assert experiment.num_qubits == 2
    assert experiment.two_qubit_gate == cirq.CX
    assert experiment.single_qubit_gate_set == [
        cirq.PhasedXZGate(
            z_exponent=z,
            x_exponent=0.5,
            axis_phase_exponent=a,
        )
        for a, z in itertools.product(np.linspace(start=0, stop=7 / 4, num=8), repeat=2)
    ]

    experiment = XEB(single_qubit_gate_set=[cirq.X])
    assert experiment.num_qubits == 2
    assert experiment.two_qubit_gate == cirq.CZ
    assert experiment.single_qubit_gate_set == [cirq.X]


@pytest.fixture
def xeb_experiment() -> XEB:
    return XEB(single_qubit_gate_set=[cirq.X, cirq.Y, cirq.Z])


def test_build_xeb_circuit() -> None:
    with patch("numpy.random.default_rng") as rng:
        xeb_experiment = XEB(single_qubit_gate_set=[cirq.X, cirq.Y, cirq.Z])
        rng.return_value.choice.side_effect = [
            np.array([[cirq.X, cirq.Y], [cirq.Z, cirq.Y], [cirq.Y, cirq.Z]]),
            np.array([[cirq.X, cirq.Z], [cirq.X, cirq.X], [cirq.Y, cirq.Y]]),
        ]
        circuits = xeb_experiment._build_circuits(num_circuits=2, cycle_depths=[2])

    assert len(circuits) == 2

    qbs = xeb_experiment.qubits
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
    assert circuits[0].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}
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
    assert circuits[1].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}


def test_xeb_analyse_results(xeb_experiment: XEB) -> None:
    xeb_experiment._samples = MagicMock()
    # Choose example data to give perfect fit with fidelity=0.95
    xeb_experiment._raw_data = pd.DataFrame(
        [
            {
                "cycle_depth": 1,
                "circuit_depth": 3,
                "sum_p(x)p(x)": 0.3,
                "sum_p(x)p^(x)": 0.95**1 * 0.3,
            },
            {
                "cycle_depth": 1,
                "circuit_depth": 3,
                "sum_p(x)p(x)": 0.5,
                "sum_p(x)p^(x)": 0.95**1 * 0.5,
            },
            {
                "cycle_depth": 5,
                "circuit_depth": 11,
                "sum_p(x)p(x)": 0.3,
                "sum_p(x)p^(x)": 0.95**5 * 0.3,
            },
            {
                "cycle_depth": 5,
                "circuit_depth": 11,
                "sum_p(x)p(x)": 0.5,
                "sum_p(x)p^(x)": 0.95**5 * 0.5,
            },
            {
                "cycle_depth": 10,
                "circuit_depth": 21,
                "sum_p(x)p(x)": 0.3,
                "sum_p(x)p^(x)": 0.95**10 * 0.3,
            },
            {
                "cycle_depth": 10,
                "circuit_depth": 21,
                "sum_p(x)p(x)": 0.5,
                "sum_p(x)p^(x)": 0.95**10 * 0.5,
            },
        ]
    )
    results = xeb_experiment.analyze_results()

    assert xeb_experiment.results.cycle_fidelity_estimate == pytest.approx(0.95)
    assert xeb_experiment.results.cycle_fidelity_estimate_std == pytest.approx(0.0, abs=1e-8)

    assert results == xeb_experiment.results

    # Call plotting function to test no errors are raised.
    xeb_experiment.plot_results()


def test_xeb_process_probabilities(xeb_experiment: XEB) -> None:
    qubits = cirq.LineQubit.range(2)

    samples = [
        XEBSample(
            raw_circuit=cirq.Circuit(
                [
                    cirq.X(qubits[0]),
                    cirq.X(qubits[1]),
                    cirq.CX(qubits[0], qubits[1]),
                    cirq.X(qubits[0]),
                    cirq.X(qubits[1]),
                    cirq.measure(qubits),
                ]
            ),
            data={"circuit_depth": 3, "num_cycles": 1, "two_qubit_gate": "CX"},
        )
    ]
    samples[0].probabilities = {"00": 0.1, "01": 0.3, "10": 0.4, "11": 0.2}

    with patch("cirq.Simulator") as mock_simulator:
        mock_simulator.return_value.simulate.return_value.final_state_vector = [0.0, 1.0, 0.0, 0.0]
        data = xeb_experiment._process_probabilities(samples)

    expected_data = pd.DataFrame(
        [
            {
                "cycle_depth": 1,
                "circuit_depth": 3,
                "p(00)": 0.0,
                "p(01)": 1.0,
                "p(10)": 0.0,
                "p(11)": 0.0,
                "p^(00)": 0.1,
                "p^(01)": 0.3,
                "p^(10)": 0.4,
                "p^(11)": 0.2,
                "sum_p(x)p(x)": 1.0,
                "sum_p(x)p^(x)": 0.3,
            }
        ]
    )
    pd.testing.assert_frame_equal(expected_data, data)


def test_xeb_process_probabilities_missing_probs(xeb_experiment: XEB) -> None:
    qubits = cirq.LineQubit.range(2)

    samples = [
        XEBSample(
            raw_circuit=cirq.Circuit(
                [
                    cirq.X(qubits[0]),
                    cirq.X(qubits[1]),
                    cirq.CX(qubits[0], qubits[1]),
                    cirq.X(qubits[0]),
                    cirq.X(qubits[1]),
                    cirq.measure(qubits),
                ]
            ),
            data={"circuit_depth": 3, "num_cycles": 1, "two_qubit_gate": "CX"},
        )
    ]

    with pytest.warns(
        UserWarning,
        match=r"1 sample\(s\) are missing `probabilities`. These samples have been omitted.",
    ):
        data = xeb_experiment._process_probabilities(samples)
    pd.testing.assert_frame_equal(data, pd.DataFrame())


def test_xebsample_sum_probs_square_no_values() -> None:
    sample = XEBSample(raw_circuit=cirq.Circuit(), data={})
    with pytest.raises(RuntimeError, match="`target_probabilities` have not yet been initialised"):
        sample.sum_target_probs_square()


def test_xebsample_sum_cross_sample_probs_no_values() -> None:
    sample = XEBSample(raw_circuit=cirq.Circuit(), data={})
    with pytest.raises(RuntimeError, match="`target_probabilities` have not yet been initialised"):
        sample.sum_target_cross_sample_probs()

    sample.target_probabilities = {"example": 0.6}
    with pytest.raises(RuntimeError, match="`sample_probabilities` have not yet been initialised"):
        sample.sum_target_cross_sample_probs()
