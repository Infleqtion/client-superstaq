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

from cirq_superstaq import compiler_output
from cirq_superstaq._version import __version__
from cirq_superstaq.compiler_output import active_qubit_indices, measured_qubit_indices
from cirq_superstaq.job import Job
from cirq_superstaq.ops import (
    AQTICCX,
    AQTITOFFOLI,
    BSWAP,
    BSWAP_INV,
    CR,
    CZ3,
    CZ3_INV,
    ZX,
    AceCR,
    AceCRMinusPlus,
    AceCRPlusMinus,
    Barrier,
    BSwapPowGate,
    ParallelGates,
    ParallelRGate,
    QubitSubspaceGate,
    QutritCZPowGate,
    QutritZ0,
    QutritZ0PowGate,
    QutritZ1,
    QutritZ1PowGate,
    QutritZ2,
    QutritZ2PowGate,
    RGate,
    ZXPowGate,
    ZZSwapGate,
    approx_eq_mod,
    barrier,
    parallel_gates_operation,
)
from cirq_superstaq.sampler import Sampler
from cirq_superstaq.serialization import (
    SUPERSTAQ_RESOLVERS,
    deserialize_circuits,
    serialize_circuits,
)
from cirq_superstaq.service import Service

__all__ = [
    "AQTICCX",
    "AQTITOFFOLI",
    "AceCR",
    "AceCRMinusPlus",
    "AceCRPlusMinus",
    "BSWAP",
    "BSWAP_INV",
    "BSwapPowGate",
    "Barrier",
    "CR",
    "CZ3",
    "CZ3_INV",
    "Job",
    "measured_qubit_indices",
    "ParallelGates",
    "ParallelRGate",
    "QubitSubspaceGate",
    "QutritCZPowGate",
    "QutritZ0",
    "QutritZ0PowGate",
    "QutritZ1",
    "QutritZ1PowGate",
    "QutritZ2",
    "QutritZ2PowGate",
    "RGate",
    "SUPERSTAQ_RESOLVERS",
    "Sampler",
    "Service",
    "ZX",
    "ZXPowGate",
    "ZZSwapGate",
    "__version__",
    "active_qubit_indices",
    "approx_eq_mod",
    "barrier",
    "compiler_output",
    "deserialize_circuits",
    "ops",
    "parallel_gates_operation",
    "serialization",
    "serialize_circuits",
]
