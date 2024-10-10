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
# mypy: disable-error-code="union-attr"
from __future__ import annotations

from unittest.mock import MagicMock, patch

import cirq
import pandas as pd
import pytest

import supermarq.qcvv.irb
from supermarq.qcvv.base_experiment import Sample
from supermarq.qcvv.irb import IRB


def test_irb_init() -> None:
    experiment = IRB()
    assert experiment.num_qubits == 1
    assert experiment.interleaved_gate == cirq.ops.SingleQubitCliffordGate.Z
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()

    experiment = IRB(interleaved_gate=cirq.ops.SingleQubitCliffordGate.X)
    assert experiment.num_qubits == 1
    assert experiment.interleaved_gate == cirq.ops.SingleQubitCliffordGate.X
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()

    experiment = IRB(interleaved_gate=cirq.CZ)
    assert experiment.num_qubits == 2
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()

    experiment = IRB(interleaved_gate=None, clifford_op_gateset=cirq.SqrtIswapTargetGateset())
    assert experiment.num_qubits == 1
    assert experiment.interleaved_gate is None
    assert experiment.clifford_op_gateset == cirq.SqrtIswapTargetGateset()


def test_irb_bad_init() -> None:
    with pytest.raises(NotImplementedError):
        IRB(interleaved_gate=None, num_qubits=3)


def test_reduce_clifford_sequence() -> None:
    sequence = [
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.Z,
    ]

    combined_gate = supermarq.qcvv.irb._reduce_clifford_seq(sequence)  # type: ignore[arg-type]
    assert combined_gate == cirq.ops.SingleQubitCliffordGate.Z


def test_random_single_qubit_clifford() -> None:
    gate = IRB().random_single_qubit_clifford()
    assert isinstance(gate, cirq.ops.SingleQubitCliffordGate)


def test_irb_random_clifford() -> None:
    exp = IRB()
    gate = exp.random_clifford()
    assert isinstance(gate, cirq.SingleQubitCliffordGate)

    exp = IRB(interleaved_gate=cirq.CZ)
    gate = exp.random_clifford()
    assert isinstance(gate, cirq.CliffordGate)
    assert gate.num_qubits() == 2


def test_gates_per_clifford() -> None:
    exp = IRB(random_seed=1)
    gates = exp.gates_per_clifford(samples=1000)
    assert gates["single_qubit_gates"] == pytest.approx(0.95, abs=0.1)
    assert gates["two_qubit_gates"] == 0.0

    exp = IRB(interleaved_gate=cirq.CZ, random_seed=1)
    gates = exp.gates_per_clifford(samples=1000)
    assert gates["single_qubit_gates"] == pytest.approx(4.5, abs=0.25)
    assert gates["two_qubit_gates"] == pytest.approx(1.5, abs=0.1)


def test_irb_process_probabilities() -> None:
    irb_experiment = IRB()
    samples = [
        Sample(
            raw_circuit=cirq.Circuit(),
            data={
                "num_cycles": 20,
                "circuit_depth": 23,
                "experiment": "example",
                "single_qubit_gates": 10,
                "two_qubit_gates": 15,
            },
        )
    ]
    samples[0].probabilities = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}

    data = irb_experiment._process_probabilities(samples)

    expected_data = pd.DataFrame(
        [
            {
                "clifford_depth": 20,
                "circuit_depth": 23,
                "experiment": "example",
                "single_qubit_gates": 10,
                "two_qubit_gates": 15,
                "00": 0.1,
                "01": 0.2,
                "10": 0.3,
                "11": 0.4,
            }
        ]
    )

    pd.testing.assert_frame_equal(expected_data, data)


def test_irb_process_probabilities_missing_probs() -> None:
    irb_experiment = IRB()
    samples = [
        Sample(
            raw_circuit=cirq.Circuit(),
            data={
                "num_cycles": 20,
                "circuit_depth": 23,
                "experiment": "example",
                "single_qubit_gates": 10,
                "two_qubit_gates": 15,
            },
        )
    ]

    with pytest.warns(
        UserWarning,
        match=r"1 sample\(s\) are missing probabilities. These samples have been omitted.",
    ):
        data = irb_experiment._process_probabilities(samples)

    expected_data = pd.DataFrame()
    pd.testing.assert_frame_equal(expected_data, data)


def test_irb_build_circuit() -> None:
    irb_experiment = IRB()
    with patch("supermarq.qcvv.irb.IRB.random_single_qubit_clifford") as mock_random_clifford:
        mock_random_clifford.side_effect = [
            cirq.ops.SingleQubitCliffordGate.Z,
            cirq.ops.SingleQubitCliffordGate.Z,
            cirq.ops.SingleQubitCliffordGate.Z,
            cirq.ops.SingleQubitCliffordGate.X,
            cirq.ops.SingleQubitCliffordGate.X,
            cirq.ops.SingleQubitCliffordGate.X,
        ]

        circuits = irb_experiment._build_circuits(2, [3])
        expected_circuits = [
            Sample(
                raw_circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "num_cycles": 3,
                    "circuit_depth": 4,
                    "experiment": "RB",
                    "single_qubit_gates": 4,
                    "two_qubit_gates": 0,
                },
            ),
            Sample(
                raw_circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "num_cycles": 3,
                    "circuit_depth": 6,
                    "experiment": "IRB",
                    "single_qubit_gates": 6,
                    "two_qubit_gates": 0,
                },
            ),
            Sample(
                raw_circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "num_cycles": 3,
                    "circuit_depth": 4,
                    "experiment": "RB",
                    "single_qubit_gates": 4,
                    "two_qubit_gates": 0,
                },
            ),
            Sample(
                raw_circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "num_cycles": 3,
                    "circuit_depth": 7,
                    "experiment": "IRB",
                    "single_qubit_gates": 7,
                    "two_qubit_gates": 0,
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


def test_analyse_results() -> None:
    irb_experiment = IRB()
    irb_experiment._samples = MagicMock()
    # Noise added to allow estimate of covariance (otherwise scipy errors)
    irb_experiment._raw_data = pd.DataFrame(
        [
            {
                "clifford_depth": 1,
                "circuit_depth": 2,
                "experiment": "RB",
                "0": 0.5 * 0.95**1 + 0.5 + 0.0000001,
                "1": 0.5 - 0.5 * 0.95**1 - 0.0000001,
            },
            {
                "clifford_depth": 1,
                "circuit_depth": 3,
                "experiment": "IRB",
                "0": 0.5 * 0.8**1 + 0.5 - 0.00000015,
                "1": 0.5 - 0.5 * 0.8**1 + 0.00000015,
            },
            {
                "clifford_depth": 2,
                "circuit_depth": 3,
                "experiment": "RB",
                "0": 0.5 * 0.95**2 + 0.5 + 0.0000011,
                "1": 0.5 - 0.5 * 0.95**2 - 0.0000011,
            },
            {
                "clifford_depth": 2,
                "circuit_depth": 4,
                "experiment": "IRB",
                "0": 0.5 * 0.8**2 + 0.5 - 0.00000017,
                "1": 0.5 - 0.5 * 0.8**2 + 0.00000017,
            },
            {
                "clifford_depth": 5,
                "circuit_depth": 6,
                "experiment": "RB",
                "0": 0.5 * 0.95**5 + 0.5 + 0.0000002,
                "1": 0.5 - 0.5 * 0.95**5 - 0.0000002,
            },
            {
                "clifford_depth": 5,
                "circuit_depth": 11,
                "experiment": "IRB",
                "0": 0.5 * 0.8**5 + 0.5 - 0.0000001,
                "1": 0.5 - 0.5 * 0.8**5 + 0.0000001,
            },
            {
                "clifford_depth": 10,
                "circuit_depth": 11,
                "experiment": "RB",
                "0": 0.5 * 0.95**10 + 0.5 + 0.00000015,
                "1": 0.5 - 0.5 * 0.95**10 - 0.00000015,
            },
            {
                "clifford_depth": 10,
                "circuit_depth": 21,
                "experiment": "IRB",
                "0": 0.5 * 0.8**10 + 0.5 + 0.00000012,
                "1": 0.5 - 0.5 * 0.8**10 - 0.00000012,
            },
        ]
    )
    irb_experiment.analyze_results()

    assert irb_experiment.results.rb_decay_coefficient == pytest.approx(0.95, abs=1e-5)
    assert irb_experiment.results.irb_decay_coefficient == pytest.approx(0.8, abs=1e-5)
    assert irb_experiment.results.average_interleaved_gate_error == pytest.approx(
        0.5 * (1 - 0.8 / 0.95), abs=1e-5
    )

    # Test that plotting results doesn't introduce any errors.
    irb_experiment.plot_results()


def test_analyse_results_rb() -> None:
    rb_experiment = IRB(interleaved_gate=None)

    rb_experiment._samples = MagicMock()
    rb_experiment._raw_data = pd.DataFrame(
        [
            {
                "clifford_depth": 1,
                "circuit_depth": 2,
                "experiment": "RB",
                "0": 0.5 * 0.95**1 + 0.5 - 0.0000001,
                "1": 0.5 - 0.5 * 0.95**1 + 0.0000001,
            },
            {
                "clifford_depth": 3,
                "circuit_depth": 4,
                "experiment": "RB",
                "0": 0.5 * 0.95**3 + 0.5 - 0.0000003,
                "1": 0.5 - 0.5 * 0.95**3 + 0.0000003,
            },
            {
                "clifford_depth": 5,
                "circuit_depth": 6,
                "experiment": "RB",
                "0": 0.5 * 0.95**5 + 0.5 - 0.0000002,
                "1": 0.5 - 0.5 * 0.95**5 + 0.0000002,
            },
            {
                "clifford_depth": 10,
                "circuit_depth": 11,
                "experiment": "RB",
                "0": 0.5 * 0.95**10 + 0.5 + 0.00000015,
                "1": 0.5 - 0.5 * 0.95**10 - 0.00000015,
            },
        ]
    )
    rb_experiment.analyze_results()

    assert rb_experiment.results.rb_decay_coefficient == pytest.approx(0.95, abs=1e-5)
    assert rb_experiment.results.average_error_per_clifford == pytest.approx(
        0.5 * (1 - 0.95), abs=1e-5
    )

    # Test that plotting results doesn't introduce any errors.
    rb_experiment.plot_results()
