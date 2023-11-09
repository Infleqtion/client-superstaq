from typing import Any, Dict, Optional, TypedDict

import pydantic

Job = TypedDict(
    "Job",
    {
        "provider_id": str,
        "num_qubits": int,
        "status": str,
        "target": str,
        "circuit_type": str,
        "compiled_circuit": str,
        "input_circuit": Optional[str],
        "data": Optional[Dict[str, Any]],
        "samples": Optional[Dict[str, int]],
        "shots": Optional[int],
        "state_vector": Optional[str],
    },
)


class Target(pydantic.BaseModel):
    """A data class to store data returned from a `/get_targets` request."""

    target: str
    supports_submit: bool = False
    supports_submit_qubo: bool = False
    supports_compile: bool = False
    available: bool = False
    retired: bool = False
