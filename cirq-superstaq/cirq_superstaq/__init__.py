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

from cirq_superstaq import compiler_output, evaluation, resource_counters, validation
from cirq_superstaq._version import __version__
from cirq_superstaq.circuits import (
    msd_5_to_1,
    msd_7_to_1,
    msd_15_to_1,
)
from cirq_superstaq.compiler_output import active_qubit_indices, measured_qubit_indices
from cirq_superstaq.job import Job, JobV3
from cirq_superstaq.ops import (
    AQTICCX,
    AQTITOFFOLI,
    BSWAP,
    BSWAP_INV,
    CR,
    CZ3,
    CZ3_INV,
    DD,
    SWAP3,
    ZX,
    AceCR,
    AceCRMinusPlus,
    AceCRPlusMinus,
    Barrier,
    BSwapPowGate,
    DDPowGate,
    ParallelGates,
    ParallelRGate,
    QubitSubspaceGate,
    QuditSwapGate,
    QutritCZPowGate,
    QutritZ0,
    QutritZ0PowGate,
    QutritZ1,
    QutritZ1PowGate,
    QutritZ2,
    QutritZ2PowGate,
    RGate,
    StrippedCZGate,
    VirtualZPowGate,
    ZXPowGate,
    ZZSwapGate,
    approx_eq_mod,
    barrier,
    parallel_gates_operation,
    qubit_subspace_op,
    qudit_swap_op,
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
    "BSWAP",
    "BSWAP_INV",
    "CR",
    "CZ3",
    "CZ3_INV",
    "DD",
    "SUPERSTAQ_RESOLVERS",
    "SWAP3",
    "ZX",
    "AceCR",
    "AceCRMinusPlus",
    "AceCRPlusMinus",
    "BSwapPowGate",
    "Barrier",
    "DDPowGate",
    "Job",
    "JobV3",
    "ParallelGates",
    "ParallelRGate",
    "QubitSubspaceGate",
    "QuditSwapGate",
    "QutritCZPowGate",
    "QutritZ0",
    "QutritZ0PowGate",
    "QutritZ1",
    "QutritZ1PowGate",
    "QutritZ2",
    "QutritZ2PowGate",
    "RGate",
    "Sampler",
    "Service",
    "StrippedCZGate",
    "VirtualZPowGate",
    "ZXPowGate",
    "ZZSwapGate",
    "__version__",
    "active_qubit_indices",
    "approx_eq_mod",
    "barrier",
    "compiler_output",
    "deserialize_circuits",
    "evaluation",
    "measured_qubit_indices",
    "msd_5_to_1",
    "msd_7_to_1",
    "msd_15_to_1",
    "parallel_gates_operation",
    "qubit_subspace_op",
    "qudit_swap_op",
    "resource_counters",
    "serialize_circuits",
    "validation",
]
