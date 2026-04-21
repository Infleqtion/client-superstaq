# Copyright 2026 Infleqtion
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

import textwrap

import general_superstaq as gss


def test_compiler_output_repr() -> None:
    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )
    qubit_map: dict[int, int] = {0: 0}
    assert (
        repr(gss.compiler_output.CompilerOutput(jaqal_program, qubit_map, qubit_map))
        == f"CompilerOutput({jaqal_program!r}, {{0: 0}}, {{0: 0}}, None, None, None)"
    )

    jaqal_programs = [jaqal_program, jaqal_program]
    assert (
        repr(
            gss.compiler_output.CompilerOutput(
                jaqal_programs, [qubit_map, qubit_map], [qubit_map, qubit_map]
            )
        )
        == f"CompilerOutput({jaqal_programs!r}, [{{0: 0}}, {{0: 0}}], [{{0: 0}}, {{0: 0}}], "
        "None, None, None)"
    )


def test_compiler_output_eq() -> None:
    jaqal_program = textwrap.dedent(
        """\
        register allqubits[2]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )
    co = gss.compiler_output.CompilerOutput(jaqal_program, {0: 0}, {0: 1})
    assert co != 1
    assert not co.jaqal_programs
    assert not co.jaqal_program

    jaqal_program_alt = ""
    assert co != gss.compiler_output.CompilerOutput(jaqal_program_alt, {}, {})

    assert (
        gss.compiler_output.CompilerOutput(
            [jaqal_program, jaqal_program], [{0: 0}, {0: 0}], [{0: 1}, {0: 1}]
        )
        != co
    )

    assert gss.compiler_output.CompilerOutput(
        [jaqal_program, jaqal_program], [{0: 0}, {0: 0}], [{0: 1}, {0: 1}]
    ) != gss.compiler_output.CompilerOutput(
        [jaqal_program, jaqal_program_alt], [{0: 0}, {}], [{0: 1}, {}]
    )
