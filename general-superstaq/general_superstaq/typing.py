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

from typing import Any, TypedDict

import pydantic


class Job(TypedDict):
    """A class to store data for a Superstaq job."""

    provider_id: str
    num_qubits: int
    status: str
    target: str
    pulse_gate_circuits: str | None
    circuit_type: str
    compiled_circuit: str
    input_circuit: str | None
    data: dict[str, Any] | None
    samples: dict[str, int] | None
    shots: int | None
    state_vector: str | None


class Target(pydantic.BaseModel):
    """A data class to store data returned from a `/get_targets` request."""

    target: str
    supports_submit: bool = False
    supports_submit_qubo: bool = False
    supports_compile: bool = False
    available: bool = False
    retired: bool = False
    accessible: bool = False
