from typing import Any, Dict, Optional, TypedDict

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
