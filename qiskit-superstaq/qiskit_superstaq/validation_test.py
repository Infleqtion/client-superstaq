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
from __future__ import annotations

import pytest
import qiskit

import qiskit_superstaq as qss


def test_validate_qiskit_circuits() -> None:
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    with pytest.raises(
        ValueError,
        match=r"Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a sequence "
        "of `qiskit.QuantumCircuit` instances.",
    ):
        qss.validation.validate_qiskit_circuits("invalid_qc_input")

    with pytest.raises(
        ValueError,
        match=r"Invalid 'circuits' input. Must be a `qiskit.QuantumCircuit` or a "
        "sequence of `qiskit.QuantumCircuit` instances.",
    ):
        qss.validation.validate_qiskit_circuits([qc, "invalid_qc_input"])
