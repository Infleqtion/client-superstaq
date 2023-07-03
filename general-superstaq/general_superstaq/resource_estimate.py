from dataclasses import InitVar, dataclass
from typing import Dict, Optional


@dataclass
class ResourceEstimate:
    """A class to store data returned from a /resource_estimate request."""

    num_single_qubit_gates: Optional[int] = None
    num_two_qubit_gates: Optional[int] = None
    depth: Optional[int] = None
    json_data: InitVar[Dict[str, int]] = None

    def __post_init__(self, json_data: Optional[Dict[str, int]] = None) -> None:
        """Initializes `ResourceEstimate` object with JSON data, if specified.

        Args:
            json_data: Optional dictionary containing JSON data from a /resource_estimate request.
        """
        if json_data is not None:
            assert "num_single_qubit_gates" in json_data
            assert "num_two_qubit_gates" in json_data
            assert "depth" in json_data

            self.num_single_qubit_gates = json_data["num_single_qubit_gates"]
            self.num_two_qubit_gates = json_data["num_two_qubit_gates"]
            self.depth = json_data["depth"]
