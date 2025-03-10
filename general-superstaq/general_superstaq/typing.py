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
