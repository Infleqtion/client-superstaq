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
# mypy: disable-error-code=method-assign
from __future__ import annotations

import os
from unittest.mock import MagicMock

import cirq
import pandas as pd
import pytest

from cirq_superstaq.qcvv.base_experiment import Sample
from cirq_superstaq.qcvv.irb import IRB


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_irb_init() -> None:
    experiment = IRB()
    assert experiment.num_qubits == 1
    assert experiment.interleaved_gate == cirq.ops.SingleQubitCliffordGate.Z

    experiment = IRB(interleaved_gate=cirq.ops.SingleQubitCliffordGate.X)
    assert experiment.num_qubits == 1
    assert experiment.interleaved_gate == cirq.ops.SingleQubitCliffordGate.X

    with pytest.raises(NotImplementedError):
        IRB(num_qubits=2)


@pytest.fixture
def irb_experiment() -> IRB:
    return IRB()


def test_reduce_clifford_sequence() -> None:
    sequence = [
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.Z,
    ]

    combined_gate = IRB._reduce_clifford_seq(sequence)
    assert combined_gate == cirq.ops.SingleQubitCliffordGate.Z


def test_random_single_qubit_clifford() -> None:
    gate = IRB._random_single_qubit_clifford()
    assert isinstance(gate, cirq.ops.SingleQubitCliffordGate)


def test_invert_clifford_circuit(irb_experiment: IRB) -> None:
    circuit = cirq.Circuit(
        [
            cirq.ops.SingleQubitCliffordGate.X(irb_experiment.qubits[0]),
            cirq.ops.SingleQubitCliffordGate.Y(irb_experiment.qubits[0]),
        ]
    )
    inverse = irb_experiment._invert_clifford_circuit(circuit)
    expected_inverse = circuit + cirq.ops.SingleQubitCliffordGate.Z(irb_experiment.qubits[0])

    cirq.testing.assert_same_circuits(inverse, expected_inverse)


def test_irb_process_probabilities(irb_experiment: IRB) -> None:

    samples = [
        Sample(
            circuit=cirq.Circuit(),
            data={
                "num_cycles": 20,
                "circuit_depth": 23,
                "experiment": "example",
            },
        )
    ]
    samples[0].probabilities = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}
    irb_experiment._samples = samples

    irb_experiment.process_probabilities()

    expected_data = pd.DataFrame(
        [
            {
                "clifford_depth": 20,
                "circuit_depth": 23,
                "experiment": "example",
                "00": 0.1,
                "01": 0.2,
                "10": 0.3,
                "11": 0.4,
            }
        ]
    )

    pd.testing.assert_frame_equal(expected_data, irb_experiment.raw_data)


def test_irb_build_circuit(irb_experiment: IRB) -> None:
    irb_experiment._random_single_qubit_clifford = (mock_random_clifford := MagicMock())
    mock_random_clifford.side_effect = [
        cirq.ops.SingleQubitCliffordGate.Z,
        cirq.ops.SingleQubitCliffordGate.Z,
        cirq.ops.SingleQubitCliffordGate.Z,
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.X,
    ]

    circuits = irb_experiment.build_circuits(2, [3])
    expected_circuits = [
        Sample(
            circuit=cirq.Circuit(
                [
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                ]
            ),
            data={
                "num_cycles": 3,
                "circuit_depth": 4,
                "experiment": "RB",
            },
        ),
        Sample(
            circuit=cirq.Circuit(
                [
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.I(*irb_experiment.qubits),
                ]
            ),
            data={
                "num_cycles": 3,
                "circuit_depth": 7,
                "experiment": "IRB",
            },
        ),
        Sample(
            circuit=cirq.Circuit(
                [
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                ]
            ),
            data={
                "num_cycles": 3,
                "circuit_depth": 4,
                "experiment": "RB",
            },
        ),
        Sample(
            circuit=cirq.Circuit(
                [
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.X(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Z(*irb_experiment.qubits),
                    cirq.ops.SingleQubitCliffordGate.Y(*irb_experiment.qubits),
                ]
            ),
            data={
                "num_cycles": 3,
                "circuit_depth": 7,
                "experiment": "IRB",
            },
        ),
    ]

    assert len(circuits) == 4

    cirq.testing.assert_same_circuits(circuits[0].circuit, expected_circuits[0].circuit)
    assert circuits[0].data == expected_circuits[0].data
    cirq.testing.assert_same_circuits(circuits[1].circuit, expected_circuits[1].circuit)
    assert circuits[1].data == expected_circuits[1].data
    cirq.testing.assert_same_circuits(circuits[2].circuit, expected_circuits[2].circuit)
    assert circuits[2].data == expected_circuits[2].data
    cirq.testing.assert_same_circuits(circuits[3].circuit, expected_circuits[3].circuit)
    assert circuits[3].data == expected_circuits[3].data


def test_analyse_results(irb_experiment: IRB) -> None:
    irb_experiment._raw_data = pd.DataFrame(
        [
            {
                "clifford_depth": 1,
                "circuit_depth": 2,
                "experiment": "RB",
                "0": 0.5 * 0.95**1 + 0.5,
                "1": 0.5 - 0.5 * 0.95**1,
            },
            {
                "clifford_depth": 1,
                "circuit_depth": 3,
                "experiment": "IRB",
                "0": 0.5 * 0.8**1 + 0.5,
                "1": 0.5 - 0.5 * 0.8**1,
            },
            {
                "clifford_depth": 5,
                "circuit_depth": 6,
                "experiment": "RB",
                "0": 0.5 * 0.95**5 + 0.5,
                "1": 0.5 - 0.5 * 0.95**5,
            },
            {
                "clifford_depth": 5,
                "circuit_depth": 11,
                "experiment": "IRB",
                "0": 0.5 * 0.8**5 + 0.5,
                "1": 0.5 - 0.5 * 0.8**5,
            },
            {
                "clifford_depth": 10,
                "circuit_depth": 11,
                "experiment": "RB",
                "0": 0.5 * 0.95**10 + 0.5,
                "1": 0.5 - 0.5 * 0.95**10,
            },
            {
                "clifford_depth": 10,
                "circuit_depth": 21,
                "experiment": "IRB",
                "0": 0.5 * 0.8**10 + 0.5,
                "1": 0.5 - 0.5 * 0.8**10,
            },
        ]
    )
    irb_experiment.analyse_results()

    assert irb_experiment.results.rb_layer_fidelity == pytest.approx(0.95)
    assert irb_experiment.results.irb_layer_fidelity == pytest.approx(0.8)
    assert irb_experiment.results.interleaved_gate_error == pytest.approx(1 - 0.8 / 0.95)

    # Test that plotting results doesn't introduce any errors.
    irb_experiment.plot_results()
