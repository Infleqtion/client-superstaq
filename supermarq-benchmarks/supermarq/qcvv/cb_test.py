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
from unittest.mock import MagicMock, patch

import cirq
import cirq.testing
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import CB, Sample


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_cb_init() -> None:
    with patch("cirq_superstaq.service.Service"):

        with pytest.raises(
            RuntimeError, match="Cannot have both `num_channels` and `pauli_channels` be None."
        ):
            qubit = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.X(qubit)])
            CB(circuit)

        with pytest.raises(
            RuntimeError, match="This cycle benchmarking is only valid for Clifford elements."
        ):  
            qubit = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.T(qubit)])
            CB(circuit, num_channels=1)

        with pytest.raises(
            RuntimeError, match="All Pauli channels must be over 1 qubits. XX is over 2 qubits."
        ):  
            qubit = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.X(qubit)])
            CB(circuit, pauli_channels=["XX"])

        with pytest.raises(
            RuntimeError, match="All Pauli channels must be over 2 qubits. Y is over 1 qubits."
        ):  
            qubits = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.X(qubits[0]), cirq.Z(qubits[1])])
            CB(circuit, pauli_channels=["Y"])
        
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.T(qubit), cirq.X(qubit), cirq.T(qubit)])

        experiment = CB(circuit, num_channels=3)
        assert experiment.num_qubits == 1
        assert len(experiment.pauli_channels) == 3
        assert experiment._dressed_measurement
        assert experiment._matrix_order == 2

        experiment = CB(circuit, num_channels=6)
        assert len(experiment.pauli_channels) == 4

        experiment = CB(circuit, num_channels=6, dressed_measurement=False)
        assert not experiment._dressed_measurement

        experiment = CB(circuit, pauli_channels=["X", "Z"])
        assert len(experiment.pauli_channels) == 2

        experiment = CB(circuit, pauli_channels=["X", "Z", "Z"])
        assert len(experiment.pauli_channels) == 2

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
                                    cirq.H(qubits[0]), 
                                    cirq.CX(qubits[0], qubits[1]),
                                    cirq.H(qubits[0])
                                ])

        experiment = CB(circuit, num_channels=5)
        assert experiment.num_qubits == 2
        assert experiment._dressed_measurement
        assert experiment._matrix_order == 2
        assert len(experiment.pauli_channels) == 5

        experiment = CB(circuit, num_channels=20)
        assert len(experiment.pauli_channels) == 16
        

@pytest.fixture
def cb_experiment() -> CB:
    with patch("cirq_superstaq.service.Service"):
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
                                    cirq.H(qubits[0]), 
                                    cirq.CX(qubits[0], qubits[1]),
                                    cirq.H(qubits[0])
                                ])
        pauli_channels = ["XY", "YZ"]
        return CB(circuit, pauli_channels=pauli_channels)


def test_build_circuits(cb_experiment: CB) -> None:
    with patch("supermarq.qcvv.xeb.random.choices") as random_choices:
        random_choices.side_effect = [
            [p1, p2] for p1, p2 in itertools.product("IXYZ", "IXYZ")
        ]
        samples = cb_experiment._build_circuits(1, [1, 2])

    assert len(samples) == 2
    qubits = cb_experiment.qubits
    cirq.testing.assert_same_circuits(
        samples[0].raw_circuit,
        cirq.Circuit(
            [
                cirq.Y(qubits[0])**0.5,
                cirq.X(qubits[1])**(-0.5),
                cirq.I(qubits[0]),
                cirq.I(qubits[1]),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.I(qubits[0]),
                cirq.X(qubits[1]),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.TaggedOperation(cirq.CX(qubits[0], qubits[1]), "no_compile"),
                cirq.TaggedOperation(cirq.H(qubits[0]), "no_compile"),
                cirq.I(qubits[0]),
                cirq.Y(qubits[1]),
                cirq.Y(qubits[0])**(-0.5),
                cirq.X(qubits[1])**0.5,
                cirq.measure(qubits)
            ]
        )
    )
    assert samples[0].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}
 


# def test_build_xeb_circuit(xeb_experiment: XEB) -> None:
#     with patch("supermarq.qcvv.xeb.random.choices") as random_choice:
#         random_choice.side_effect = [
#             [cirq.X, cirq.Y],
#             [cirq.Z, cirq.Y],
#             [cirq.Y, cirq.Z],
#             [cirq.X, cirq.Z],
#             [cirq.X, cirq.X],
#             [cirq.Y, cirq.Y],
#         ]
#         circuits = xeb_experiment._build_circuits(num_circuits=2, cycle_depths=[2])

#     assert len(circuits) == 2

#     qbs = xeb_experiment.qubits
#     cirq.testing.assert_same_circuits(
#         circuits[0].circuit,
#         cirq.Circuit(
#             [
#                 cirq.X(qbs[0]),
#                 cirq.Y(qbs[1]),
#                 cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
#                 cirq.Z(qbs[0]),
#                 cirq.Y(qbs[1]),
#                 cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
#                 cirq.Y(qbs[0]),
#                 cirq.Z(qbs[1]),
#                 cirq.measure(qbs),
#             ]
#         ),
#     )
#     assert circuits[0].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}
#     cirq.testing.assert_same_circuits(
#         circuits[1].circuit,
#         cirq.Circuit(
#             [
#                 cirq.X(qbs[0]),
#                 cirq.Z(qbs[1]),
#                 cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
#                 cirq.X(qbs[0]),
#                 cirq.X(qbs[1]),
#                 cirq.TaggedOperation(cirq.CZ(*qbs), "no_compile"),
#                 cirq.Y(qbs[0]),
#                 cirq.Y(qbs[1]),
#                 cirq.measure(qbs),
#             ]
#         ),
#     )
#     assert circuits[1].data == {"circuit_depth": 5, "num_cycles": 2, "two_qubit_gate": "CZ"}


# def test_xeb_analyse_results(xeb_experiment: XEB) -> None:
#     xeb_experiment._samples = MagicMock()
#     # Choose example data to give perfect fit with fidelity=0.95
#     xeb_experiment._raw_data = pd.DataFrame(
#         [
#             {
#                 "cycle_depth": 1,
#                 "circuit_depth": 3,
#                 "sum_p(x)p(x)": 0.3,
#                 "sum_p(x)p^(x)": 0.95**1 * 0.3,
#             },
#             {
#                 "cycle_depth": 1,
#                 "circuit_depth": 3,
#                 "sum_p(x)p(x)": 0.5,
#                 "sum_p(x)p^(x)": 0.95**1 * 0.5,
#             },
#             {
#                 "cycle_depth": 5,
#                 "circuit_depth": 11,
#                 "sum_p(x)p(x)": 0.3,
#                 "sum_p(x)p^(x)": 0.95**5 * 0.3,
#             },
#             {
#                 "cycle_depth": 5,
#                 "circuit_depth": 11,
#                 "sum_p(x)p(x)": 0.5,
#                 "sum_p(x)p^(x)": 0.95**5 * 0.5,
#             },
#             {
#                 "cycle_depth": 10,
#                 "circuit_depth": 21,
#                 "sum_p(x)p(x)": 0.3,
#                 "sum_p(x)p^(x)": 0.95**10 * 0.3,
#             },
#             {
#                 "cycle_depth": 10,
#                 "circuit_depth": 21,
#                 "sum_p(x)p(x)": 0.5,
#                 "sum_p(x)p^(x)": 0.95**10 * 0.5,
#             },
#         ]
#     )
#     results = xeb_experiment.analyze_results()

#     assert xeb_experiment.results.cycle_fidelity_estimate == pytest.approx(0.95)
#     assert xeb_experiment.results.cycle_fidelity_estimate_std == pytest.approx(0.0, abs=1e-8)

#     assert results == xeb_experiment.results

#     # Call plotting function to test no errors are raised.
#     xeb_experiment.plot_results()


# def test_xeb_process_probabilities(xeb_experiment: XEB) -> None:
#     qubits = cirq.LineQubit.range(2)

#     samples = [
#         XEBSample(
#             raw_circuit=cirq.Circuit(
#                 [
#                     cirq.X(qubits[0]),
#                     cirq.X(qubits[1]),
#                     cirq.CX(qubits[0], qubits[1]),
#                     cirq.X(qubits[0]),
#                     cirq.X(qubits[1]),
#                     cirq.measure(qubits),
#                 ]
#             ),
#             data={"circuit_depth": 3, "num_cycles": 1, "two_qubit_gate": "CX"},
#         )
#     ]
#     samples[0].probabilities = {"00": 0.1, "01": 0.3, "10": 0.4, "11": 0.2}

#     with patch("cirq.Simulator") as mock_simulator:
#         mock_simulator.return_value.simulate.return_value.final_state_vector = [0.0, 1.0, 0.0, 0.0]
#         data = xeb_experiment._process_probabilities(samples)

#     expected_data = pd.DataFrame(
#         [
#             {
#                 "cycle_depth": 1,
#                 "circuit_depth": 3,
#                 "p(00)": 0.0,
#                 "p(01)": 1.0,
#                 "p(10)": 0.0,
#                 "p(11)": 0.0,
#                 "p^(00)": 0.1,
#                 "p^(01)": 0.3,
#                 "p^(10)": 0.4,
#                 "p^(11)": 0.2,
#                 "sum_p(x)p(x)": 1.0,
#                 "sum_p(x)p^(x)": 0.3,
#             }
#         ]
#     )
#     pd.testing.assert_frame_equal(expected_data, data)


# def test_xebsample_sum_probs_square_no_values() -> None:
#     sample = XEBSample(raw_circuit=cirq.Circuit(), data={})
#     with pytest.raises(RuntimeError, match="`target_probabilities` have not yet been initialised"):
#         sample.sum_target_probs_square()


# def test_xebsample_sum_cross_sample_probs_no_values() -> None:
#     sample = XEBSample(raw_circuit=cirq.Circuit(), data={})
#     with pytest.raises(RuntimeError, match="`target_probabilities` have not yet been initialised"):
#         sample.sum_target_cross_sample_probs()

#     sample.target_probabilities = {"example": 0.6}
#     with pytest.raises(RuntimeError, match="`sample_probabilities` have not yet been initialised"):
#         sample.sum_target_cross_sample_probs()
