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

from dataclasses import InitVar, dataclass


@dataclass
class ResourceEstimate:
    """A class to store data returned from a /resource_estimate request."""

    num_single_qubit_gates: int | None = None
    num_two_qubit_gates: int | None = None
    depth: int | None = None
    json_data: InitVar[dict[str, int] | None] = None

    def __post_init__(self, json_data: dict[str, int] | None) -> None:
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
