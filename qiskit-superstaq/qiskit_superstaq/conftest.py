from typing import Dict

import general_superstaq
import pytest

my_dict = {
    "target_info": {
        "num_qubits": 4,
        "target": "cq_hilbert_simulator",
        "coupling_map": [[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 3], [3, 1], [3, 2]],
        "supports_midcircuit_measurement": None,
        "native_gate_set": ["cz", "gr", "rz"],
        "max_experiments": None,
        "max_shots": 2048,
        "processor_type": None,
        "open_pulse": False,
        "conditional": False,
    }
}


@pytest.fixture()
def mock_target_info() -> Dict[str, object]:
    """Initializes mock Qiskit Runtime sampler fixture."""
    return my_dict
