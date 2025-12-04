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
# mypy: disable-error-code="union-attr"
from __future__ import annotations

import pathlib
import re
from unittest.mock import patch

import cirq
import cirq_superstaq as css
import pandas as pd
import pytest

import supermarq.qcvv.irb
from supermarq.qcvv.base_experiment import Sample
from supermarq.qcvv.irb import IRB, IRBResults, RBResults


@pytest.fixture
def irb() -> IRB:
    return IRB(num_circuits=10, cycle_depths=[1, 3, 5])


def test_irb_init() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)

    experiment = IRB(num_circuits=10, cycle_depths=[1, 3, 5])
    assert experiment.num_qubits == 1
    assert experiment.qubits == (q0,)
    assert experiment.interleaved_gate == cirq.Z
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()
    assert experiment.num_circuits == 10
    assert experiment.cycle_depths == [1, 3, 5]

    experiment = IRB(num_circuits=10, cycle_depths=[1, 3, 5], interleaved_gate=cirq.X)
    assert experiment.num_qubits == 1
    assert experiment.qubits == (q0,)
    assert experiment.interleaved_gate == cirq.X
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()
    assert experiment.num_circuits == 10
    assert experiment.cycle_depths == [1, 3, 5]

    experiment = IRB(num_circuits=10, cycle_depths=[1, 3, 5], interleaved_gate=cirq.CZ)
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q0, q1)
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()
    assert experiment.num_circuits == 10
    assert experiment.cycle_depths == [1, 3, 5]
    assert experiment.interleaved_gate == cirq.CZ

    experiment = IRB(num_circuits=10, cycle_depths=[1, 3, 5], interleaved_gate=cirq.CZ(q1, q2))
    assert experiment.num_qubits == 2
    assert experiment.qubits == (q1, q2)
    assert experiment.clifford_op_gateset == cirq.CZTargetGateset()
    assert experiment.num_circuits == 10
    assert experiment.cycle_depths == [1, 3, 5]
    assert all(sample.circuit.all_qubits() == {q1, q2} for sample in experiment.samples)
    assert experiment.interleaved_gate == cirq.CZ

    experiment = IRB(
        num_circuits=10,
        cycle_depths=[1, 3, 5],
        interleaved_gate=None,
        clifford_op_gateset=cirq.SqrtIswapTargetGateset(),
    )
    assert experiment.num_qubits == 1
    assert experiment.qubits == (q0,)
    assert experiment.interleaved_gate is None
    assert experiment.clifford_op_gateset == cirq.SqrtIswapTargetGateset()
    assert experiment.num_circuits == 10
    assert experiment.cycle_depths == [1, 3, 5]


def test_irb_bad_init() -> None:
    with pytest.raises(NotImplementedError):
        IRB(
            interleaved_gate=None,
            qubits=3,
            num_circuits=10,
            cycle_depths=[1, 3, 5],
        )

    with pytest.raises(ValueError, match=r"must be a Clifford"):
        IRB(1, [2], interleaved_gate=cirq.T)


def test_reduce_clifford_sequence() -> None:
    sequence = [
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.X,
        cirq.ops.SingleQubitCliffordGate.Z,
    ]

    combined_gate = supermarq.qcvv.irb._reduce_clifford_seq(sequence)  # type: ignore[arg-type]
    assert combined_gate == cirq.ops.SingleQubitCliffordGate.Z


def test_random_single_qubit_clifford(irb: IRB) -> None:
    gate = irb.random_single_qubit_clifford()
    assert isinstance(gate, cirq.ops.SingleQubitCliffordGate)


def test_irb_random_clifford(irb: IRB) -> None:
    gate = irb.random_clifford()
    assert isinstance(gate, cirq.SingleQubitCliffordGate)

    exp = IRB(num_circuits=10, cycle_depths=[1, 5, 10], interleaved_gate=cirq.CZ)
    gate = exp.random_clifford()
    assert isinstance(gate, cirq.CliffordGate)
    assert gate.num_qubits() == 2


def test_gates_per_clifford() -> None:
    exp = IRB(random_seed=1, num_circuits=10, cycle_depths=[1, 5, 10])
    gates = exp.gates_per_clifford(samples=1000)
    assert gates["single_qubit_gates"] == pytest.approx(0.95, abs=0.1)
    assert gates["two_qubit_gates"] == 0.0

    exp = IRB(interleaved_gate=cirq.CZ, random_seed=1, num_circuits=10, cycle_depths=[1, 5, 10])
    gates = exp.gates_per_clifford(samples=1000)
    assert gates["single_qubit_gates"] == pytest.approx(4.5, abs=0.25)
    assert gates["two_qubit_gates"] == pytest.approx(1.5, abs=0.1)


def test_irb_build_circuit() -> None:
    irb_experiment = IRB(num_circuits=10, cycle_depths=[1, 5, 10])
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
                circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "clifford_depth": 3,
                    "circuit_depth": 4,
                    "experiment": "RB",
                    "single_qubit_gates": 4,
                    "two_qubit_gates": 0,
                },
                circuit_realization=1,
            ),
            Sample(
                circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "clifford_depth": 3,
                    "circuit_depth": 6,
                    "experiment": "IRB",
                    "single_qubit_gates": 6,
                    "two_qubit_gates": 0,
                },
                circuit_realization=2,
            ),
            Sample(
                circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "clifford_depth": 3,
                    "circuit_depth": 4,
                    "experiment": "RB",
                    "single_qubit_gates": 4,
                    "two_qubit_gates": 0,
                },
                circuit_realization=3,
            ),
            Sample(
                circuit=cirq.Circuit(
                    [
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=0.0)(
                            *irb_experiment.qubits
                        ),
                        css.barrier(*irb_experiment.qubits),
                        cirq.TaggedOperation(cirq.Z(*irb_experiment.qubits), "no_compile"),
                        css.barrier(*irb_experiment.qubits),
                        cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=1.0, z_exponent=1.0)(
                            *irb_experiment.qubits
                        ),
                        cirq.measure(irb_experiment.qubits),
                    ]
                ),
                data={
                    "clifford_depth": 3,
                    "circuit_depth": 7,
                    "experiment": "IRB",
                    "single_qubit_gates": 7,
                    "two_qubit_gates": 0,
                },
                circuit_realization=4,
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
    irb_results = IRBResults(
        target="example", experiment=IRB(num_circuits=1, cycle_depths=[1, 2, 5, 10])
    )
    # Noise added to allow estimate of covariance (otherwise scipy errors)
    irb_results.data = pd.DataFrame(
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
    irb_results.analyze()

    assert irb_results.rb_decay_coefficient == pytest.approx(0.95, abs=1e-5)
    assert irb_results.irb_decay_coefficient == pytest.approx(0.8, abs=1e-5)
    assert irb_results.rb_decay_coefficient_std == pytest.approx(0.0, abs=1e-5)
    assert irb_results.irb_decay_coefficient_std == pytest.approx(0.0, abs=1e-5)
    assert irb_results.average_interleaved_gate_error == pytest.approx(
        0.5 * (1 - 0.8 / 0.95), abs=1e-5
    )

    # Test that plotting results doesn't introduce any errors.
    irb_results.plot_results()


def test_analyse_results_rb() -> None:
    rb_results = RBResults(
        target="example",
        experiment=IRB(interleaved_gate=None, num_circuits=1, cycle_depths=[1, 3, 5, 10]),
    )
    # Noise added to allow estimate of covariance (otherwise scipy errors)

    rb_results.data = pd.DataFrame(
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
    rb_results.analyze()

    assert rb_results.rb_decay_coefficient == pytest.approx(0.95, abs=1e-5)
    assert rb_results.average_error_per_clifford == pytest.approx(0.5 * (1 - 0.95), abs=1e-5)

    # Test that plotting results doesn't introduce any errors.
    rb_results.plot_results()


def test_analyse_results_plot_saving(tmp_path: pathlib.Path) -> None:
    irb_results = IRBResults(
        target="example", experiment=IRB(num_circuits=1, cycle_depths=[1, 2, 5, 10])
    )
    # Noise added to allow estimate of covariance (otherwise scipy errors)
    irb_results.data = pd.DataFrame(
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
    filename = tmp_path / "example_filename.png"
    irb_results.analyze(plot_filename=filename.as_posix())
    assert pathlib.Path(filename).exists()


def test_analyse_results_rb_plot_saving(tmp_path: pathlib.Path) -> None:
    rb_results = RBResults(
        target="example",
        experiment=IRB(interleaved_gate=None, num_circuits=1, cycle_depths=[1, 3, 5, 10]),
    )
    # Noise added to allow estimate of covariance (otherwise scipy errors)

    rb_results.data = pd.DataFrame(
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
    filename = tmp_path / "example_filename.png"
    rb_results.analyze(plot_filename=filename.as_posix())
    assert pathlib.Path(filename).exists()


def test_results_no_data() -> None:
    results = IRBResults(target="example", experiment=IRB(1, []))
    with pytest.raises(RuntimeError, match=r"No data stored. Cannot perform fit."):
        results._fit_decay()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot make plot."):
        results._plot_results()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot make plot."):
        results.plot_results()

    rb_results = RBResults(target="example", experiment=IRB(1, []))
    with pytest.raises(RuntimeError, match=r"No data stored. Cannot perform fit."):
        rb_results._fit_decay()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot make plot."):
        rb_results._plot_results()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot make plot."):
        rb_results.plot_results()


def test_irb_results_not_analyzed() -> None:
    results = IRBResults(target="example", experiment=IRB(1, []))
    for attr in [
        "rb_decay_coefficient",
        "rb_decay_coefficient_std",
        "irb_decay_coefficient",
        "irb_decay_coefficient_std",
        "average_interleaved_gate_error",
        "average_interleaved_gate_error_std",
    ]:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
        ):
            getattr(results, attr)


def test_rb_results_not_analyzed() -> None:
    results = RBResults(target="example", experiment=IRB(1, []))
    for attr in [
        "rb_decay_coefficient",
        "rb_decay_coefficient_std",
        "average_error_per_clifford",
        "average_error_per_clifford_std",
    ]:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
        ):
            getattr(results, attr)


def test_dump_and_load(
    tmp_path_factory: pytest.TempPathFactory,
    irb: IRB,
) -> None:
    filename = tmp_path_factory.mktemp("tempdir") / "file.json"
    irb.to_file(filename)
    exp = IRB.from_file(filename)

    assert exp.samples == irb.samples
    assert exp.num_qubits == irb.num_qubits
    assert exp.num_circuits == irb.num_circuits
    assert exp.cycle_depths == irb.cycle_depths
    assert exp.interleaved_gate == irb.interleaved_gate
    assert exp.clifford_op_gateset == irb.clifford_op_gateset

    # Set interleaved gate to None and check again
    irb.interleaved_gate = None
    irb.to_file(filename)
    exp = IRB.from_file(filename)

    assert exp.samples == irb.samples
    assert exp.num_qubits == irb.num_qubits
    assert exp.num_circuits == irb.num_circuits
    assert exp.cycle_depths == irb.cycle_depths
    assert exp.interleaved_gate == irb.interleaved_gate
    assert exp.clifford_op_gateset == irb.clifford_op_gateset
