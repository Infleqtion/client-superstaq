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

import os
import pathlib
import re
import sys
import textwrap
import uuid
from unittest.mock import MagicMock, patch

import cirq
import numpy as np
import pandas as pd
import pytest

from supermarq.qcvv import SU2, SU2Results


@pytest.fixture(scope="session", autouse=True)
def patch_tqdm() -> None:
    os.environ["TQDM_DISABLE"] = "1"


def test_su2_init() -> None:
    with patch("cirq_superstaq.service.Service"):
        experiment = SU2(4, [1, 2, 3, 4], cirq.CNOT)
    assert experiment.num_qubits == 2
    assert experiment.two_qubit_gate == cirq.CNOT
    assert experiment.num_circuits == 4
    assert experiment.cycle_depths == [1, 2, 3, 4]


def test_su2_init_raises() -> None:
    with (
        patch("cirq_superstaq.service.Service"),
        pytest.raises(
            ValueError,
            match=r"The `two_qubit_gate` parameter must be a gate that acts on exactly two qubits.",
        ),
    ):
        SU2(4, [1, 2, 3, 4], cirq.X)


@pytest.fixture
def su2_experiment() -> SU2:
    with patch("cirq_superstaq.service.Service"):
        return SU2(4, [1, 2, 3, 4], cirq.CNOT)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Python version < 3.10")
def test_build_circuits(su2_experiment: SU2) -> None:  # pragma: no cover
    # By extension also tests the _component() method
    # Patch the PhasedXZ drawing function to ignore the rotation parameters
    cirq.PhasedXZGate._circuit_diagram_info_ = MagicMock(
        return_value=cirq.CircuitDiagramInfo(wire_symbols=["PhXZ"])
    )

    samples = su2_experiment._build_circuits(2, [1, 3])
    assert len(samples) == 4
    for sample in samples:
        assert "num_two_qubit_gates" in sample.data
        assert sample.data["num_two_qubit_gates"] in [2, 6]
        if sample.data["num_two_qubit_gates"] == 2:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───│───@[no_compile]───│───X───│───@[no_compile]───│───PhXZ───│───X───│───PhXZ───│───X───│───PhXZ───M───
                                 │   │               │       │   │               │          │       │          │       │          │
                    1: ───PhXZ───│───X───────────────│───X───│───X───────────────│───PhXZ───│───X───│───PhXZ───│───X───│───PhXZ───M───
                    """  # noqa: E501
                ),
            )
        else:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───│───@[no_compile]───│───X───│───@[no_compile]───│───PhXZ───│───@[no_compile]───│───X───│───@[no_compile]───│───PhXZ───│───@[no_compile]───│───X───│───@[no_compile]───│───PhXZ───M───
                                 │   │               │       │   │               │          │   │               │       │   │               │          │   │               │       │   │               │          │
                    1: ───PhXZ───│───X───────────────│───X───│───X───────────────│───PhXZ───│───X───────────────│───X───│───X───────────────│───PhXZ───│───X───────────────│───X───│───X───────────────│───PhXZ───M───
                    """  # noqa: E501
                ),
            )


@pytest.mark.skipif(sys.version_info >= (3, 10), reason="Python version >= 3.10")
def test_build_circuits_old(su2_experiment: SU2) -> None:  # pragma: no cover
    # By extension also tests the _component() method
    # Patch the PhasedXZ drawing function to ignore the rotation parameters
    cirq.PhasedXZGate._circuit_diagram_info_ = MagicMock(
        return_value=cirq.CircuitDiagramInfo(wire_symbols=["PhXZ"])
    )

    samples = su2_experiment._build_circuits(2, [1, 3])
    assert len(samples) == 4
    for sample in samples:
        assert "num_two_qubit_gates" in sample.data
        assert sample.data["num_two_qubit_gates"] in [2, 6]
        if sample.data["num_two_qubit_gates"] == 2:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───│───@['no_compile']───│───X───│───@['no_compile']───│───PhXZ───│───X───│───PhXZ───│───X───│───PhXZ───M───
                                 │   │                 │       │   │                 │          │       │          │       │          │
                    1: ───PhXZ───│───X─────────────────│───X───│───X─────────────────│───PhXZ───│───X───│───PhXZ───│───X───│───PhXZ───M───
                    """  # noqa: E501
                ),
            )
        else:
            cirq.testing.assert_has_diagram(
                sample.circuit,
                textwrap.dedent(
                    """
                    0: ───PhXZ───│───@['no_compile']───│───X───│───@['no_compile']───│───PhXZ───│───@['no_compile']───│───X───│───@['no_compile']───│───PhXZ───│───@['no_compile']───│───X───│───@['no_compile']───│───PhXZ───M───
                                 │   │                 │       │   │                 │          │   │                 │       │   │                 │          │   │                 │       │   │                 │          │
                    1: ───PhXZ───│───X─────────────────│───X───│───X─────────────────│───PhXZ───│───X─────────────────│───X───│───X─────────────────│───PhXZ───│───X─────────────────│───X───│───X─────────────────│───PhXZ───M───
                    """  # noqa: E501
                ),
            )


def test_analyse_results(tmp_path: pathlib.Path, su2_experiment: SU2) -> None:
    def decay(x: float) -> float:
        return (3 * 0.75 * 0.975**x + 1) / 4

    result = SU2Results(
        target="example",
        experiment=su2_experiment,
        data=pd.DataFrame(
            [
                {
                    "num_two_qubit_gates": 2,
                    "circuit_realization": 1,
                    "uuid": uuid.uuid4(),
                    "00": decay(2),
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1 - decay(2),
                },
                {
                    "num_two_qubit_gates": 4,
                    "circuit_realization": 1,
                    "uuid": uuid.uuid4(),
                    "00": decay(4),
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1 - decay(4),
                },
                {
                    "num_two_qubit_gates": 6,
                    "circuit_realization": 1,
                    "uuid": uuid.uuid4(),
                    "00": decay(6),
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1 - decay(6),
                },
                {
                    "num_two_qubit_gates": 8,
                    "circuit_realization": 1,
                    "uuid": uuid.uuid4(),
                    "00": decay(8),
                    "01": 0.0,
                    "10": 0.0,
                    "11": 1 - decay(8),
                },
            ]
        ),
    )

    plot_filename = tmp_path / "example.png"

    result.analyze(plot_filename=plot_filename.as_posix())

    assert pathlib.Path(tmp_path / "example.png").exists()

    assert result.two_qubit_gate_fidelity == pytest.approx(0.975)
    assert result.two_qubit_gate_fidelity_std == pytest.approx(0.0)
    assert result.two_qubit_gate_error == pytest.approx(0.025)
    assert result.two_qubit_gate_error_std == pytest.approx(0.0)
    assert result.single_qubit_noise == pytest.approx(0.25)
    assert result.single_qubit_noise_std == pytest.approx(0.0)


def test_haar_random_rotation() -> None:
    r = SU2._haar_random_rotation()
    assert isinstance(r, cirq.Gate)
    assert r.num_qubits() == 1

    with patch("cirq.testing.random_special_unitary", return_value=np.eye(2)):
        assert SU2._haar_random_rotation() == cirq.I


def test_result_not_analyzed() -> None:
    result = SU2Results(target="example", experiment=MagicMock(spec=SU2))

    for attr in [
        "two_qubit_gate_fidelity",
        "two_qubit_gate_fidelity_std",
        "two_qubit_gate_error",
        "two_qubit_gate_error_std",
        "single_qubit_noise",
        "single_qubit_noise_std",
    ]:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Value has not yet been estimated. Please run `.analyze()` method."),
        ):
            _ = getattr(result, attr)


def test_result_missing_data() -> None:
    result = SU2Results(target="example", experiment=MagicMock(spec=SU2))

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot perform analysis."):
        result._analyze()

    with pytest.raises(RuntimeError, match=r"No data stored. Cannot plot results."):
        result.plot_results()


def test_dump_and_load(
    tmp_path_factory: pytest.TempPathFactory,
    su2_experiment: SU2,
) -> None:
    filename = tmp_path_factory.mktemp("tempdir") / "file.json"
    su2_experiment.to_file(filename)
    exp = SU2.from_file(filename)

    assert exp.samples == su2_experiment.samples
    assert exp.num_qubits == su2_experiment.num_qubits
    assert exp.num_circuits == su2_experiment.num_circuits
    assert exp.cycle_depths == su2_experiment.cycle_depths
    assert exp.two_qubit_gate == su2_experiment.two_qubit_gate
