# Copyright 2021 The Cirq Developers
#
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

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

import cirq
import cirq_superstaq as css
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample

mock_plot = MagicMock()
mock_print = MagicMock()


@dataclass
class ExampleResults(QCVVResults):
    """Example results class for testing"""

    _example_final_result: float | None = None

    def _analyze(self) -> None:
        self._example_final_result = 3.142

    def plot_results(self) -> None:
        mock_plot()

    def print_results(self) -> None:
        mock_print("This is a test")

    @property
    def example_final_result(self) -> float:
        if self._example_final_result is None:
            raise self._not_analyzed
        return self._example_final_result


class ExampleExperiment(QCVVExperiment[ExampleResults]):
    """Example experiment class for testing"""

    def _build_circuits(self, num_circuits: int, cycle_depths: Iterable[int]) -> Sequence[Sample]:
        return [
            Sample(
                circuit=MagicMock(spec=cirq.Circuit), data={"num": k, "depth": d}, circuit_index=k
            )
            for k in range(num_circuits)
            for d in cycle_depths
        ]


@pytest.fixture
def abc_experiment() -> ExampleExperiment:
    with patch("supermarq.qcvv.base_experiment.QCVVExperiment._validate_circuits"):
        return ExampleExperiment(
            num_qubits=2,
            num_circuits=10,
            cycle_depths=[1, 3, 5],
            random_seed=42,
            results_cls=ExampleResults,
            service_details="Some other details",
        )


@pytest.fixture
def sample_circuits() -> list[Sample]:
    qubits = cirq.LineQubit.range(2)
    return [
        Sample(
            circuit=cirq.Circuit(cirq.CZ(*qubits), cirq.CZ(*qubits), cirq.measure(*qubits)),
            data={"circuit": 1},
            circuit_index=1,
        ),
        Sample(
            circuit=cirq.Circuit(cirq.CX(*qubits), cirq.measure(*qubits)),
            data={"circuit": 2},
            circuit_index=2,
        ),
    ]


def test_qcvv_experiment_init(
    abc_experiment: ExampleExperiment,
) -> None:
    assert abc_experiment.num_qubits == 2
    assert abc_experiment.num_circuits == 10
    assert abc_experiment.cycle_depths == [1, 3, 5]
    assert abc_experiment._results_cls == ExampleResults
    assert abc_experiment._service_kwargs == {"service_details": "Some other details"}
    assert len(abc_experiment.samples) == 30
    assert isinstance(abc_experiment._rng, np.random.Generator)


def test_results_init(
    abc_experiment: ExampleExperiment,
) -> None:
    results = ExampleResults(
        target="target", experiment=abc_experiment, job=MagicMock(spec=css.Job)
    )
    assert results.target == "target"
    assert results.samples == abc_experiment.samples
    assert results.num_circuits == 10
    assert results.num_qubits == 2


def test_experiment_init_with_bad_layers() -> None:
    with pytest.raises(
        ValueError, match="The `cycle_depths` iterator can only include positive values."
    ):
        ExampleExperiment(
            num_qubits=2,
            num_circuits=10,
            cycle_depths=[0],
            random_seed=42,
            results_cls=ExampleResults,
            service_details="Some other details",
        )


def test_results_not_analyzed(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(
        target="target", experiment=abc_experiment, job=MagicMock(spec=css.Job)
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
    ):
        _ = results.example_final_result


def test_results_job_still_running(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(
        target="target", experiment=abc_experiment, job=MagicMock(spec=css.Job)
    )
    results.job.status.return_value = "Pending"  # type: ignore[union-attr]
    with pytest.warns(
        Warning,
        match=(
            "Experiment data is not yet ready to analyse. This is likely because "
            "the Superstaq job has not yet been completed. Either wait and try again "
            "later, or interrogate the `.job` attribute."
        ),
    ):
        results.analyze()


def test_results_job_no_data(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(
        target="target",
        experiment=abc_experiment,
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "No data available and no Superstaq job to use to collect data. Please manually add "
            "results data in order to perform analysis"
        ),
    ):
        results.analyze()


def test_results_analyze(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(
        target="target", experiment=abc_experiment, data=MagicMock(spec=pd.DataFrame)
    )

    results.analyze(plot_results=True, print_results=True)
    assert results.example_final_result == 3.142
    mock_plot.assert_called_once()
    mock_print.assert_called_once_with("This is a test")


def test_results_ready(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(
        target="target", experiment=abc_experiment, data=MagicMock(spec=pd.DataFrame)
    )
    assert results.data_ready


def test_results_ready_from_job(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    results = ExampleResults(
        target="target", experiment=abc_experiment, job=MagicMock(spec=css.Job)
    )
    results.job.status.return_value = "Done"  # type: ignore[union-attr]
    results.job.counts.return_value = [  # type: ignore[union-attr]
        {
            "00": 20,
            "01": 5,
            "11": 10,
        },
        {
            "00": 30,
            "01": 5,
        },
    ]
    assert results.data_ready
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {"circuit": 1, "00": 20 / 35, "01": 5 / 35, "10": 0.0, "11": 10 / 35},
                {"circuit": 2, "00": 30 / 35, "01": 5 / 35, "10": 0.0, "11": 0.0},
            ]
        ),
        check_like=True,
    )


def test_run_with_simulator(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    cirq.measurement_key_name = MagicMock()
    abc_experiment.samples = sample_circuits
    test_sim = MagicMock()
    mock_result = MagicMock()
    mock_result.histogram.return_value = {0: 0, 1: 100, 2: 0, 3: 0}
    test_sim.run.return_value = mock_result

    results = abc_experiment.run_with_simulator(simulator=test_sim, repetitions=100)

    # Test simulator calls
    test_sim.run.assert_has_calls(
        [
            call(sample_circuits[0].circuit, repetitions=100),
            call(sample_circuits[1].circuit, repetitions=100),
        ],
        any_order=True,
    )

    assert results.experiment == abc_experiment
    assert results.target == "local_simulator"

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {"circuit_index": 1, "circuit": 1, "00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0},
                {"circuit_index": 2, "circuit": 2, "00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0},
            ]
        ),
    )


def test_run_with_simulator_default_target(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    cirq.measurement_key_name = MagicMock()
    cirq.Simulator = (target := MagicMock())  # type: ignore [misc]
    abc_experiment.samples = sample_circuits
    mock_result = MagicMock()
    mock_result.histogram.return_value = {0: 0, 1: 100, 2: 0, 3: 0}
    target().run.return_value = mock_result

    results = abc_experiment.run_with_simulator(repetitions=100)

    # Test simulator calls
    target().run.assert_has_calls(
        [
            call(sample_circuits[0].circuit, repetitions=100),
            call(sample_circuits[1].circuit, repetitions=100),
        ],
        any_order=True,
    )

    assert results.experiment == abc_experiment
    assert results.target == "local_simulator"

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {"circuit_index": 1, "circuit": 1, "00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0},
                {"circuit_index": 2, "circuit": 2, "00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0},
            ]
        ),
    )


def test_run_on_device(abc_experiment: ExampleExperiment, sample_circuits: list[Sample]) -> None:
    abc_experiment.samples = sample_circuits

    with patch("cirq_superstaq.Service") as mock_service:
        results = abc_experiment.run_on_device(
            target="example_target", repetitions=100, **{"some": "options"}
        )

    mock_service.return_value.create_job.assert_called_once_with(
        [sample_circuits[0].circuit, sample_circuits[1].circuit],
        target="example_target",
        method=None,
        repetitions=100,
        some="options",
    )

    assert results.job == mock_service.return_value.create_job.return_value
    assert results.target == "example_target"
    assert results.experiment == abc_experiment


def test_run_on_device_dry_run(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits

    with patch("cirq_superstaq.Service") as mock_service:
        results = abc_experiment.run_on_device(
            target="example_target", repetitions=100, method="dry-run"
        )

    mock_service.return_value.create_job.assert_called_once_with(
        [sample_circuits[0].circuit, sample_circuits[1].circuit],
        target="example_target",
        method="dry-run",
        repetitions=100,
    )
    assert results.job == mock_service.return_value.create_job.return_value
    assert results.target == "example_target"
    assert results.experiment == abc_experiment


def test_interleave_circuit() -> None:
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(*[cirq.X(qubit) for _ in range(4)])

    # With last gate
    interleaved_circuit = QCVVExperiment._interleave_op(circuit, cirq.Z(qubit), include_final=True)
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
        ),
    )

    # Without last gate
    interleaved_circuit = QCVVExperiment._interleave_op(circuit, cirq.Z(qubit), include_final=False)
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
            cirq.TaggedOperation(cirq.Z(qubit), "no_compile"),
            cirq.X(qubit),
        ),
    )


def test_validate_circuits(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    # Should't get any errors with the base circuits
    abc_experiment._validate_circuits(sample_circuits)

    # Add a gate so not all measurements are terminal
    sample_circuits[0].circuit += cirq.X(abc_experiment.qubits[0])
    with pytest.raises(
        ValueError, match="QCVV experiment circuits can only contain terminal measurements."
    ):
        abc_experiment._validate_circuits(sample_circuits)

    # Remove measurements
    sample_circuits[0].circuit = sample_circuits[0].circuit[:-2] + cirq.measure(
        abc_experiment.qubits[0]
    )
    with pytest.raises(
        ValueError,
        match="The terminal measurement in QCVV experiment circuits must measure all qubits.",
    ):
        abc_experiment._validate_circuits(sample_circuits)

    # Remove all measurements
    sample_circuits[0].circuit = sample_circuits[0].circuit[:-2]
    with pytest.raises(
        ValueError,
        match="QCVV experiment circuits must contain measurements.",
    ):
        abc_experiment._validate_circuits(sample_circuits)


def test_run_with_callable(
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    abc_experiment.samples = sample_circuits
    test_callable = MagicMock()
    test_callable.return_value = {"01": 0.2, "10": 0.7, "11": 0.1}

    results = abc_experiment.run_with_callable(test_callable, some="kwargs")

    test_callable.assert_has_calls(
        [
            call(sample_circuits[0].circuit, some="kwargs"),
            call(sample_circuits[1].circuit, some="kwargs"),
        ]
    )

    assert results.target == "callable"
    assert results.experiment == abc_experiment

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {"circuit": 1, "00": 0.0, "01": 0.2, "10": 0.7, "11": 0.1},
                {"circuit": 2, "00": 0.0, "01": 0.2, "10": 0.7, "11": 0.1},
            ]
        ),
        check_like=True,
    )


def test_run_with_callable_mixd_keys(
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    abc_experiment.samples = sample_circuits
    test_callable = MagicMock()
    test_callable.return_value = {1: 0.2, "10": 0.7, 3: 0.1}

    results = abc_experiment.run_with_callable(test_callable, some="kwargs")

    test_callable.assert_has_calls(
        [
            call(sample_circuits[0].circuit, some="kwargs"),
            call(sample_circuits[1].circuit, some="kwargs"),
        ]
    )

    assert results.target == "callable"
    assert results.experiment == abc_experiment

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {"circuit": 1, "00": 0.0, "01": 0.2, "10": 0.7, "11": 0.1},
                {"circuit": 2, "00": 0.0, "01": 0.2, "10": 0.7, "11": 0.1},
            ]
        ),
        check_like=True,
    )


def test_run_with_callable_bad_bitstring(
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    abc_experiment.samples = sample_circuits
    test_callable = MagicMock()
    test_callable.return_value = {"000": 0.0, "01": 0.2, "10": 0.8}

    with pytest.raises(
        ValueError,
        match=("The key contains the wrong number of bits. Got 3 entries " "but expected 2 bits."),
    ):
        abc_experiment.run_with_callable(test_callable, some="kwargs")


def test_run_with_callable_bad_probabilities(
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    abc_experiment.samples = sample_circuits
    test_callable = MagicMock()
    test_callable.return_value = {"00": 0.0, "01": 0.2, "10": 0.7, "11": 0.09}

    with pytest.raises(RuntimeError, match="Provided probabilities do not sum to 1.0."):
        abc_experiment.run_with_callable(test_callable, some="kwargs")


def test_results_collect_device_counts(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    results = ExampleResults(
        target="example_target", experiment=abc_experiment, job=MagicMock(spec=css.Job)
    )
    results.job.counts.return_value = [  # type: ignore[union-attr]
        {
            "00": 20,
            "01": 5,
            "11": 10,
        },
        {
            "00": 30,
            "01": 5,
        },
    ]

    df = results._collect_device_counts()

    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            [
                {"circuit": 1, "00": 20 / 35, "01": 5 / 35, "10": 0.0, "11": 10 / 35},
                {"circuit": 2, "00": 30 / 35, "01": 5 / 35, "10": 0.0, "11": 0.0},
            ]
        ),
        check_like=True,
    )


def test_results_collect_device_counts_no_job() -> None:
    results = ExampleResults(target="example_target", experiment=MagicMock(), job=None)
    with pytest.raises(
        ValueError,
        match=("No Superstaq job associated with these results. Cannot collect device counts."),
    ):
        results._collect_device_counts()


def test_canonicalize_bitstring() -> None:
    assert QCVVExperiment._canonicalize_bitstring("00", 2) == "00"
    assert QCVVExperiment._canonicalize_bitstring(1, 2) == "01"
    assert QCVVExperiment._canonicalize_bitstring(5, 4) == "0101"

    with pytest.raises(ValueError, match="The key must be positive. Instead got -2."):
        QCVVExperiment._canonicalize_bitstring(-2, 4)

    with pytest.raises(
        ValueError,
        match=(
            "The key is too large to be encoded with 4 qubits. Got 72 " "but expected less than 16."
        ),
    ):
        QCVVExperiment._canonicalize_bitstring(72, 4)

    with pytest.raises(
        ValueError,
        match=("The key contains the wrong number of bits. Got 5 entries " "but expected 4 bits."),
    ):
        QCVVExperiment._canonicalize_bitstring("01010", 4)

    with pytest.raises(ValueError, match="All entries in the bitstring must be 0 or 1. Got 1234."):
        QCVVExperiment._canonicalize_bitstring("1234", 4)


def test_canonicalize_probabilities() -> None:
    p: dict[str | int, float] = {3: 0.3, 0: 0.1, "10": 0.6}
    assert QCVVExperiment._canonicalize_probabilities(p, 2) == {
        "00": 0.1,
        "01": 0.0,
        "10": 0.6,
        "11": 0.3,
    }

    with pytest.raises(
        RuntimeError,
        match="Provided probabilities do not sum to 1.0. Got",  # Ignore exact value due to rounding
    ):
        p[0] = 0.0
        QCVVExperiment._canonicalize_probabilities(p, 2)
