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

from cirq_superstaq import compiler_output, serialization
from cirq_superstaq._version import __version__
from cirq_superstaq.custom_gates import (
    AQTICCX,
    AQTITOFFOLI,
    CR,
    ZX,
    AceCR,
    AceCRMinusPlus,
    AceCRPlusMinus,
    Barrier,
    ParallelGates,
    ParallelRGate,
    RGate,
    ZXPowGate,
    ZZSwapGate,
    barrier,
)
from cirq_superstaq.job import Job
from cirq_superstaq.sampler import Sampler
from cirq_superstaq.service import Service

__all__ = [
    "__version__",
    "AceCR",
    "AceCRMinusPlus",
    "AceCRPlusMinus",
    "barrier",
    "Barrier",
    "compiler_output",
    "CR",
    "AQTICCX",
    "AQTITOFFOLI",
    "Job",
    "ParallelGates",
    "ParallelRGate",
    "serialization",
    "RGate",
    "Sampler",
    "Service",
    "ZX",
    "ZXPowGate",
    "ZZSwapGate",
]
