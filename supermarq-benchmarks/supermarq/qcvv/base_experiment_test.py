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
# mypy: disable-error-code=method-assign

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import cirq
import cirq_superstaq as css
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv.base_experiment import QCVVExperiment, QCVVResults, Sample, qcvv_resolver


def test_qcvv_resolver() -> None:
    assert qcvv_resolver("bad_name") is None
    assert qcvv_resolver("supermarq.qcvv.Sample") == Sample
    assert qcvv_resolver("supermarq.qcvv.QCVVExperiment") == QCVVExperiment

    # Check for something that is not explicitly exported
    assert qcvv_resolver("supermarq.qcvv.base_experiment.qcvv_resolver") is None


@dataclass
class ExampleResults(QCVVResults):
    """Example results class for testing."""

    _example_final_result: float | None = None

    def _analyze(self) -> None:
        self._example_final_result = 3.142

    def plot_results(self, filename: str | None = None) -> plt.Figure:
        fig = plt.Figure()
        if filename:
            fig.savefig(filename)
        return fig

    def print_results(self) -> None:
        print("This is a test")  # noqa: T201

    @property
    def example_final_result(self) -> float:
        if self._example_final_result is None:
            raise self._not_analyzed
        return self._example_final_result


class ExampleExperiment(QCVVExperiment[ExampleResults]):
    """Example experiment class for testing."""

    def __init__(
        self,
        qubits: int | Sequence[cirq.Qid],
        num_circuits: int,
        cycle_depths: Iterable[int],
        *,
        random_seed: int | None = None,
        _samples: list[Sample] | None = None,
        **kwargs: str | bool,
    ) -> None:
        super().__init__(
            qubits,
            num_circuits,
            cycle_depths,
            random_seed=random_seed,
            results_cls=ExampleResults,
            _samples=_samples,
            **kwargs,
        )

    def _build_circuits(self, num_circuits: int, cycle_depths: Iterable[int]) -> Sequence[Sample]:
        return [
            Sample(
                circuit=cirq.Circuit(cirq.measure(*self.qubits)),
                data={"depth": d},
                circuit_realization=k,
            )
            for k in range(num_circuits)
            for d in cycle_depths
        ]

    def _json_dict_(self) -> dict[str, Any]:
        return super()._json_dict_()


@pytest.fixture
def abc_experiment() -> ExampleExperiment:
    return ExampleExperiment(
        qubits=2,
        num_circuits=10,
        cycle_depths=[1, 3, 5],
        random_seed=42,
        service_details="Some other details",
    )


@pytest.fixture
def sample_circuits() -> list[Sample]:
    qubits = cirq.LineQubit.range(2)
    return [
        Sample(
            circuit=cirq.Circuit(
                cirq.X(qubits[1]), cirq.CZ(*qubits), cirq.CZ(*qubits), cirq.measure(*qubits)
            ),
            data={"circuit": 1},
            circuit_realization=1,
        ),
        Sample(
            circuit=cirq.Circuit(cirq.X(qubits[0]), cirq.CX(*qubits), cirq.measure(*qubits)),
            data={"circuit": 2},
            circuit_realization=2,
        ),
    ]


def test_qcvv_experiment_init(
    abc_experiment: ExampleExperiment,
) -> None:
    assert abc_experiment.num_qubits == 2
    assert abc_experiment.qubits == (cirq.q(0), cirq.q(1))
    assert abc_experiment.num_circuits == 10
    assert abc_experiment.cycle_depths == [1, 3, 5]
    assert abc_experiment._results_cls == ExampleResults
    assert abc_experiment._service_kwargs == {"service_details": "Some other details"}
    assert len(abc_experiment.samples) == 30
    assert isinstance(abc_experiment._rng, np.random.Generator)
    assert abc_experiment.circuits == [sample.circuit for sample in abc_experiment]

    new_experiment = ExampleExperiment(
        qubits=[cirq.q(1), cirq.q(3), cirq.q(7)],
        num_circuits=10,
        cycle_depths=[1, 3, 5],
    )
    assert new_experiment.num_qubits == 3
    assert new_experiment.qubits == (cirq.q(1), cirq.q(3), cirq.q(7))
    assert new_experiment.circuits == [sample.circuit for sample in new_experiment]


def test_results_init(
    abc_experiment: ExampleExperiment,
) -> None:
    results = ExampleResults(target="target", experiment=abc_experiment)
    assert results.target == "target"
    assert results.samples == abc_experiment.samples
    assert results.num_circuits == 10
    assert results.num_qubits == 2
    assert results.qubits == (cirq.q(0), cirq.q(1))


def test_results_getitem(
    abc_experiment: ExampleExperiment,
) -> None:
    q0, q1 = abc_experiment.qubits

    results = ExampleResults(
        target="example",
        experiment=abc_experiment,
        data=pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "00": 0.5,
                    "01": 0.5,
                    "10": 0.0,
                    "11": 0.0,
                }
                for sample in abc_experiment.samples
            ]
        ),
    )

    results_q0 = results[q0]
    assert results_q0.qubits == (q0,)
    assert results_q0.parent is results
    pd.testing.assert_frame_equal(
        results_q0.data,
        pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "0": 1.0,
                    "1": 0.0,
                }
                for sample in abc_experiment.samples
            ]
        ),
        check_like=True,
    )

    results_q1 = results[q1]
    assert results_q1.qubits == (q1,)
    assert results_q1.parent is results
    pd.testing.assert_frame_equal(
        results_q1.data,
        pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "0": 0.5,
                    "1": 0.5,
                }
                for sample in abc_experiment.samples
            ]
        ),
        check_like=True,
    )

    results_q0q1 = results[q0, q1]
    assert results_q0q1.qubits == (q0, q1)
    assert results_q0q1.parent is results
    pd.testing.assert_frame_equal(results_q0q1.data, results.data)

    results_q1q0 = results[q1, q0]
    assert results_q1q0.qubits == (q1, q0)
    assert results_q1q0.parent is results
    pd.testing.assert_frame_equal(
        results_q1q0.data,
        pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "00": 0.5,
                    "01": 0.0,
                    "10": 0.5,
                    "11": 0.0,
                }
                for sample in abc_experiment.samples
            ]
        ),
        check_like=True,
    )

    mock_job = MagicMock(spec=css.Job)
    mock_job.status.return_value = "Queued"
    results = ExampleResults(target="target", experiment=abc_experiment, job=mock_job)
    with pytest.raises(ValueError, match=r"No results to split."):
        _ = results[q0]


def test_experiment_init_with_bad_layers() -> None:
    with pytest.raises(
        ValueError, match=r"The `cycle_depths` iterator can only include positive values."
    ):
        ExampleExperiment(
            qubits=2,
            num_circuits=10,
            cycle_depths=[0],
            random_seed=42,
            service_details="Some other details",
        )


def test_results_not_analyzed(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(target="target", experiment=abc_experiment)
    with pytest.raises(
        RuntimeError,
        match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
    ):
        _ = results.example_final_result


def test_results_job_still_running(abc_experiment: ExampleExperiment) -> None:
    mock_job = MagicMock(spec=css.Job)
    mock_job.status.return_value = "Pending"
    results = ExampleResults(target="target", experiment=abc_experiment, job=mock_job)
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
            r"No data available and no Superstaq job to use to collect data. Please manually add "
            "results data in order to perform analysis"
        ),
    ):
        results.analyze()


def test_results_analyze(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(target="target", experiment=abc_experiment, data=pd.DataFrame())

    with (
        patch("matplotlib.pyplot.Figure.savefig") as mock_plot,
        patch("builtins.print") as mock_print,
    ):
        results.analyze(plot_results=True, print_results=True, plot_filename="test_name")

    assert results.example_final_result == 3.142
    mock_plot.assert_called_once_with("test_name")
    mock_print.assert_called_once_with("This is a test")


def test_results_ready(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(target="target", experiment=abc_experiment, data=pd.DataFrame())
    assert results.data_ready


def test_results_ready_from_job(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    mock_job = MagicMock(spec=css.Job)
    mock_job.status.return_value = "Done"
    mock_job.counts.return_value = [
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
    results = ExampleResults(target="target", experiment=abc_experiment, job=mock_job)
    assert results.data_ready
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {
                    "uuid": abc_experiment.samples[0].uuid,
                    "circuit_realization": 1,
                    "circuit": 1,
                    "00": 20 / 35,
                    "01": 5 / 35,
                    "10": 0.0,
                    "11": 10 / 35,
                },
                {
                    "uuid": abc_experiment.samples[1].uuid,
                    "circuit_realization": 2,
                    "circuit": 2,
                    "00": 30 / 35,
                    "01": 5 / 35,
                    "10": 0.0,
                    "11": 0.0,
                },
            ]
        ),
        check_like=True,
    )


def test_run_with_simulator(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits

    simulator = cirq.DensityMatrixSimulator()
    with patch("cirq.Simulator") as mock_sim:  # Mock default simulator to make sure it isn't called
        results = abc_experiment.run_with_simulator(simulator=simulator, repetitions=100)
        mock_sim.assert_not_called()

    assert results.experiment == abc_experiment
    assert results.target == "local_simulator"

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {
                    "uuid": abc_experiment.samples[0].uuid,
                    "circuit_realization": 1,
                    "circuit": 1,
                    "00": 0.0,
                    "01": 1.0,
                    "10": 0.0,
                    "11": 0.0,
                },
                {
                    "uuid": abc_experiment.samples[1].uuid,
                    "circuit_realization": 2,
                    "circuit": 2,
                    "00": 0.0,
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1.0,
                },
            ]
        ),
    )


def test_run_with_simulator_default_target(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits

    results = abc_experiment.run_with_simulator(repetitions=100)

    assert results.experiment == abc_experiment
    assert results.target == "local_simulator"

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {
                    "uuid": abc_experiment.samples[0].uuid,
                    "circuit_realization": 1,
                    "circuit": 1,
                    "00": 0.0,
                    "01": 1.0,
                    "10": 0.0,
                    "11": 0.0,
                },
                {
                    "uuid": abc_experiment.samples[1].uuid,
                    "circuit_realization": 2,
                    "circuit": 2,
                    "00": 0.0,
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1.0,
                },
            ]
        ),
    )


def test_run_on_device(abc_experiment: ExampleExperiment, sample_circuits: list[Sample]) -> None:
    abc_experiment.samples = sample_circuits

    with patch("cirq_superstaq.Service") as mock_service:
        results = abc_experiment.run_on_device(
            target="example_target", repetitions=100, some="options"
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


def test_interleave_circuit(abc_experiment: ExampleExperiment) -> None:
    qubits = abc_experiment.qubits
    circuit = cirq.Circuit(*[cirq.X(qubits[0]) for _ in range(4)])

    # With last gate
    interleaved_circuit = abc_experiment._interleave_layer(
        circuit, cirq.Z(qubits[0]), include_final=True
    )
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
        ),
    )

    # Without last gate
    interleaved_circuit = abc_experiment._interleave_layer(
        circuit, cirq.Z(qubits[0]), include_final=False
    )
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
        ),
    )

    # Multi-gate layer
    layer = cirq.Moment(cirq.Z(qubits[0]), cirq.H(qubits[1]))
    interleaved_circuit = abc_experiment._interleave_layer(
        circuit, layer=layer, include_final=False
    )
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            cirq.H(qubits[1]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            cirq.H(qubits[1]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.Z(qubits[0]).with_tags("no_compile"),
            cirq.H(qubits[1]).with_tags("no_compile"),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
        ),
    )

    # Empty layer
    interleaved_circuit = abc_experiment._interleave_layer(circuit, layer=None, include_final=False)
    cirq.testing.assert_same_circuits(
        interleaved_circuit,
        cirq.Circuit(
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
            css.barrier(*qubits),
            cirq.X(qubits[0]),
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
        ValueError, match=r"QCVV experiment circuits can only contain terminal measurements."
    ):
        abc_experiment._validate_circuits(sample_circuits)

    # Remove measurements
    sample_circuits[0].circuit = sample_circuits[0].circuit[:-2] + cirq.measure(
        abc_experiment.qubits[0]
    )
    with pytest.raises(
        ValueError,
        match=r"The terminal measurement in QCVV experiment circuits must measure all qubits.",
    ):
        abc_experiment._validate_circuits(sample_circuits)

    # Remove all measurements
    sample_circuits[0].circuit = sample_circuits[0].circuit[:-2]
    with pytest.raises(
        ValueError,
        match=r"QCVV experiment circuits must contain measurements.",
    ):
        abc_experiment._validate_circuits(sample_circuits)


def test_run_with_callable(abc_experiment: ExampleExperiment) -> None:
    def _example_callable(sample: Sample, some: str) -> dict[str, float]:
        assert sample
        assert some == "kwargs"
        return {"01": 0.2, "10": 0.7, "11": 0.1}

    results = abc_experiment.run_with_callable(
        _example_callable,  # type: ignore[arg-type]
        some="kwargs",
    )

    assert results.target == "callable"
    assert results.experiment == abc_experiment

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "00": 0.0,
                    "01": 0.2,
                    "10": 0.7,
                    "11": 0.1,
                }
                for sample in abc_experiment.samples
            ]
        ),
        check_like=True,
    )


def test_run_with_callable_mixed_keys(abc_experiment: ExampleExperiment) -> None:
    def _example_callable(sample: Sample, some: str) -> dict[str | int, float]:
        assert sample
        assert some == "kwargs"
        return {1: 0.2, "10": 0.7, 3: 0.1}

    results = abc_experiment.run_with_callable(
        _example_callable,  # type: ignore[arg-type]
        some="kwargs",
    )

    assert results.target == "callable"
    assert results.experiment == abc_experiment

    # Check the data is stored
    pd.testing.assert_frame_equal(
        results.data,
        pd.DataFrame(
            [
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "00": 0.0,
                    "01": 0.2,
                    "10": 0.7,
                    "11": 0.1,
                }
                for sample in abc_experiment.samples
            ]
        ),
        check_like=True,
    )


def test_run_with_callable_bad_bitstring(abc_experiment: ExampleExperiment) -> None:
    def _example_callable(sample: Sample, some: str) -> dict[str, float]:
        assert sample
        assert some == "kwargs"
        return {"000": 0.0, "01": 0.2, "10": 0.8}

    with pytest.raises(
        ValueError,
        match=(r"The key contains the wrong number of bits. Got 3 entries but expected 2 bits."),
    ):
        abc_experiment.run_with_callable(_example_callable, some="kwargs")  # type: ignore[arg-type]


def test_results_collect_device_counts(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits

    mock_job = MagicMock(spec=css.Job)
    mock_job.counts.return_value = [
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
    results = ExampleResults(target="example_target", experiment=abc_experiment, job=mock_job)

    df = results._collect_device_counts()

    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            [
                {
                    "uuid": abc_experiment.samples[0].uuid,
                    "circuit_realization": 1,
                    "circuit": 1,
                    "00": 20 / 35,
                    "01": 5 / 35,
                    "10": 0.0,
                    "11": 10 / 35,
                },
                {
                    "uuid": abc_experiment.samples[1].uuid,
                    "circuit_realization": 2,
                    "circuit": 2,
                    "00": 30 / 35,
                    "01": 5 / 35,
                    "10": 0.0,
                    "11": 0.0,
                },
            ]
        ),
        check_like=True,
    )


def test_results_collect_device_counts_no_job(abc_experiment: ExampleExperiment) -> None:
    results = ExampleResults(target="example_target", experiment=abc_experiment, job=None)
    with pytest.raises(
        ValueError,
        match=(r"No Superstaq job associated with these results. Cannot collect device counts."),
    ):
        results._collect_device_counts()


def test_results_from_records(abc_experiment: ExampleExperiment) -> None:
    # All accepted types
    records_1 = {s.uuid: {"01": 1, "10": 3} for s in abc_experiment.samples}
    records_2 = {s.uuid: {"01": 0.25, "10": 0.75} for s in abc_experiment.samples}
    records_3 = {s.uuid: {1: 1, 2: 3} for s in abc_experiment.samples}
    records_4 = {s.uuid: {1: 0.25, 2: 0.75} for s in abc_experiment.samples}

    for record in (records_1, records_2, records_3, records_4):
        results = abc_experiment.results_from_records(record)
        pd.testing.assert_frame_equal(
            results.data,
            pd.DataFrame(
                {
                    "uuid": sample.uuid,
                    "circuit_realization": sample.circuit_realization,
                    "depth": sample.data["depth"],
                    "00": 0.0,
                    "01": 0.25,
                    "10": 0.75,
                    "11": 0.0,
                }
                for sample in abc_experiment.samples
            ),
        )


def test_results_from_records_bad_input(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    # Warn for missing samples
    with pytest.warns(
        UserWarning,
        match=re.escape(
            f"The following samples are missing records: {sample_circuits[1].uuid!s}. These "
            "will not be included in the results."
        ),
    ):
        abc_experiment.results_from_records({sample_circuits[0].uuid: {"00": 10}})

    # Warn for spurious records
    new_uuid = uuid.uuid4()
    with pytest.warns(
        UserWarning,
        match=re.escape("Unable to find matching sample for 1 record(s)."),
    ):
        abc_experiment.results_from_records(
            {
                new_uuid: {"00": 10},
                sample_circuits[0].uuid: {"00": 10},
                sample_circuits[1].uuid: {"00": 10},
            }
        )

    # Error when processing samples
    with pytest.raises(ValueError, match=re.escape("No non-zero counts.")):
        abc_experiment.results_from_records(
            {
                sample_circuits[0].uuid: {"00": 0},
                sample_circuits[1].uuid: {"00": 0},
            }
        )


def test_canonicalize_bitstring() -> None:
    assert QCVVExperiment.canonicalize_bitstring("00", 2) == "00"
    assert QCVVExperiment.canonicalize_bitstring(1, 2) == "01"
    assert QCVVExperiment.canonicalize_bitstring(5, 4) == "0101"

    with pytest.raises(ValueError, match=r"The key must be positive. Instead got -2."):
        QCVVExperiment.canonicalize_bitstring(-2, 4)

    with pytest.raises(
        ValueError,
        match=(
            r"The key is too large to be encoded with 4 qubits. Got 72 but expected less than 16."
        ),
    ):
        QCVVExperiment.canonicalize_bitstring(72, 4)

    with pytest.raises(
        ValueError,
        match=(r"The key contains the wrong number of bits. Got 5 entries but expected 4 bits."),
    ):
        QCVVExperiment.canonicalize_bitstring("01010", 4)

    with pytest.raises(ValueError, match=r"All entries in the bitstring must be 0 or 1. Got 1234."):
        QCVVExperiment.canonicalize_bitstring("1234", 4)

    with pytest.raises(TypeError, match=r"Key must either be `numbers.Integral` or `str`."):
        QCVVExperiment.canonicalize_bitstring(3.141, 4)  # type: ignore[arg-type]


def test_canonicalize_probabilities() -> None:
    p1 = {0: 0.1, 1: 0.6, 3: 0.3}
    p2 = {0: 1, 1: 6, 3: 3}
    p3 = {"00": 0.1, "01": 0.6, "11": 0.3}
    p4 = {"00": 1, "01": 6, "11": 3}
    p_list: list[dict[str, int] | dict[str, float] | dict[int, int] | dict[int, float]] = [
        p1,
        p2,
        p3,
        p4,
    ]
    for p in p_list:
        assert QCVVExperiment.canonicalize_probabilities(p, 2) == {
            "00": 0.1,
            "01": 0.6,
            "10": 0.0,
            "11": 0.3,
        }

    # Test for empty dictionary
    assert QCVVExperiment.canonicalize_probabilities({}, 2) == {}


def test_canonicalize_probabilities_bad_input() -> None:
    # Negative counts
    with pytest.raises(ValueError, match=r"Probabilities/counts must be positive."):
        QCVVExperiment.canonicalize_probabilities({0: -2}, 2)

    # No non-zero counts
    with pytest.raises(ValueError, match=r"No non-zero counts."):
        QCVVExperiment.canonicalize_probabilities({0: 0, 1: 0}, 2)

    # Negative probabilities
    with pytest.raises(ValueError, match=r"Probabilities/counts must be positive."):
        QCVVExperiment.canonicalize_probabilities({0: 0.0, 1: -0.5}, 2)


def test_experiment_get_item(
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    abc_experiment.samples = sample_circuits

    for k, _ in enumerate(sample_circuits):
        assert abc_experiment[k] == sample_circuits[k]
        assert abc_experiment[sample_circuits[k].uuid] == sample_circuits[k]
        assert abc_experiment[str(sample_circuits[k].uuid)] == sample_circuits[k]

    with pytest.raises(TypeError, match=r"Key must be int, str or uuid.UUID"):
        _ = abc_experiment[3.141]  # type: ignore[index]

    with pytest.raises(
        KeyError, match=re.escape("No sample found with UUID b55adabc-39c4-4f7b-a84d-906adaf0897e")
    ):
        _ = abc_experiment["b55adabc-39c4-4f7b-a84d-906adaf0897e"]

    with pytest.raises(
        RuntimeError, match=r"Multiple samples found with matching key. Something has gone wrong."
    ):
        # Manually set duplicate sample uuids
        abc_experiment.samples[0].uuid = abc_experiment.samples[1].uuid
        _ = abc_experiment[sample_circuits[0].uuid]


def test_map_records_to_samples(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits

    records = {
        0: {0: 0.1, 1: 0.6, 3: 0.3},
        sample_circuits[1].uuid: {0: 4, 1: 6, 3: 2},
    }
    mapped_samples = abc_experiment._map_records_to_samples(records)  # type: ignore[arg-type]
    assert mapped_samples == {
        sample_circuits[0]: {0: 0.1, 1: 0.6, 3: 0.3},
        sample_circuits[1]: {0: 4, 1: 6, 3: 2},
    }


def test_map_records_to_samples_missing_key(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    with (
        pytest.warns(
            UserWarning, match=re.escape("Unable to find matching sample for 1 record(s).")
        ),
        pytest.warns(
            UserWarning,
            match=(
                f"The following samples are missing records: {sample_circuits[0].uuid}. "
                "These will not be included in the results."
            ),
        ),
    ):
        abc_experiment._map_records_to_samples(
            {
                5: {0: 0.1, 1: 0.6, 3: 0.3},
                sample_circuits[1].uuid: {0: 4, 1: 6, 3: 2},
            }
        )


def test_map_records_to_samples_bad_key_type(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    # Bad key type
    with pytest.raises(
        TypeError,
        match=re.escape("Key must be int, str or uuid.UUID, not <class 'float'>"),
    ):
        abc_experiment._map_records_to_samples(
            {
                5.0: {0: 0.1, 1: 0.6, 3: 0.3},  # type: ignore[dict-item]
                sample_circuits[1].uuid: {0: 4, 1: 6, 3: 2},
            }
        )


def test_map_records_to_samples_duplicate_keys(
    abc_experiment: ExampleExperiment, sample_circuits: list[Sample]
) -> None:
    abc_experiment.samples = sample_circuits
    with pytest.raises(
        KeyError,
        match=re.escape(
            f"Duplicate records found for sample with uuid: {sample_circuits[1].uuid!s}."
        ),
    ):
        abc_experiment._map_records_to_samples(
            {
                1: {0: 0.1, 1: 0.6, 3: 0.3},
                sample_circuits[1].uuid: {0: 4, 1: 6, 3: 2},
            }
        )


@patch("supermarq.qcvv.base_experiment.qcvv_resolver")
def test_dump_and_load(
    mock_resolver: MagicMock,
    tmp_path_factory: pytest.TempPathFactory,
    abc_experiment: ExampleExperiment,
    sample_circuits: list[Sample],
) -> None:
    temp_resolver = {
        "supermarq.qcvv.Sample": Sample,
        "supermarq.qcvv.ExampleExperiment": ExampleExperiment,
    }
    mock_resolver.side_effect = lambda x: temp_resolver.get(x, qcvv_resolver(x))

    filename = tmp_path_factory.mktemp("tempdir") / "file.json"
    abc_experiment.samples = sample_circuits
    abc_experiment.to_file(filename)
    exp = ExampleExperiment.from_file(filename)

    assert exp.samples == abc_experiment.samples
    assert exp.num_qubits == abc_experiment.num_qubits
    assert exp.num_circuits == abc_experiment.num_circuits
    assert exp.cycle_depths == abc_experiment.cycle_depths


def test_count_non_barrier_gates() -> None:
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        css.barrier(*qubits),
        cirq.CX(*qubits[0:2]),
        css.barrier(*qubits),
        css.ParallelRGate(0, 0, num_copies=4).on(*qubits),
    )
    assert ExampleExperiment._count_non_barrier_gates(circuit) == 4
    assert ExampleExperiment._count_non_barrier_gates(circuit, num_qubits=1) == 2
    assert ExampleExperiment._count_non_barrier_gates(circuit, num_qubits=2) == 1
    assert ExampleExperiment._count_non_barrier_gates(circuit, num_qubits=4) == 1
