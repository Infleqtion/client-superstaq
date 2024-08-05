# pylint: disable=missing-function-docstring,missing-class-docstring
# Copyright 2024 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from general_superstaq import ResourceEstimate


def test_resource_estimate() -> None:
    json_data = {"num_single_qubit_gates": 1, "num_two_qubit_gates": 2, "depth": 3}
    expected_re = ResourceEstimate(1, 2, 3)
    constructed_re = ResourceEstimate(json_data=json_data)

    assert repr(expected_re) == repr(constructed_re)
