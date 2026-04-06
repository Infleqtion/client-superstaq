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

import json
import textwrap

import general_superstaq as gss


def test_read_json_jaqal() -> None:
    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )
    jaqal_program_as_subcircuits = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )

    json_dict: dict[str, str | list[str]] = {
        "jaqal_strs": json.dumps([jaqal_program]),
        "initial_logical_to_physicals": json.dumps([[(0, 1)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
    }

    out = gss.compiler_output.read_json_jaqal(json_dict, circuits_is_list=False)
    assert out.circuit == jaqal_program
    assert out.initial_logical_to_physical == {0: 1}
    assert out.final_logical_to_physical == {0: 13}
    assert out.jaqal_program == jaqal_program
    assert out.jaqal_programs == [jaqal_program]

    json_dict = {
        "jaqal_strs": json.dumps([jaqal_program, jaqal_program]),
        "initial_logical_to_physicals": json.dumps([[(0, 1)], [(0, 1)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13)]]),
    }

    out = gss.compiler_output.read_json_jaqal(json_dict, circuits_is_list=False)
    # If input was a single subcircuit Jaqal, then output should be a subcircuit
    assert out.circuits == [jaqal_program_as_subcircuits]
    assert out.initial_logical_to_physicals == [{0: 1}, {0: 1}]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]
    assert out.jaqal_program == jaqal_program_as_subcircuits
    assert out.jaqal_programs == [jaqal_program_as_subcircuits]

    out = gss.compiler_output.read_json_jaqal(json_dict, circuits_is_list=True)
    # If input was a list of Jaqal, then output should be a list of Jaqal
    assert out.circuits == [jaqal_program, jaqal_program]
    assert out.initial_logical_to_physicals == [{0: 1}, {0: 1}]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]
    assert out.jaqal_programs == [jaqal_program, jaqal_program]
    assert out.jaqal_program == jaqal_program_as_subcircuits

    out = gss.compiler_output.read_json_jaqal(json_dict, circuits_is_list=True, num_eca_circuits=1)
    assert out.circuits == [[jaqal_program], [jaqal_program]]
    assert out.initial_logical_to_physicals == [[{0: 1}], [{0: 1}]]
    assert out.final_logical_to_physicals == [[{0: 13}], [{0: 13}]]
    assert out.jaqal_programs == [jaqal_program, jaqal_program]

    out = gss.compiler_output.read_json_jaqal(json_dict, circuits_is_list=True, num_eca_circuits=2)
    assert out.circuits == [[jaqal_program, jaqal_program]]
    assert out.initial_logical_to_physicals == [[{0: 1}, {0: 1}]]
    assert out.final_logical_to_physicals == [[{0: 13}, {0: 13}]]
    assert out.jaqal_programs == [jaqal_program_as_subcircuits]
