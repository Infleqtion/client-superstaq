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

import importlib
import importlib.util
import json
import pickle
import textwrap
from unittest import mock

import pytest

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
    qasm_program = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg meas[2];
        h q[0];
        cx q[0],q[1];
        barrier q[0],q[1];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        """
    )
    qasm_co = gss.compiler_output.CompilerOutput(qasm_program, {0: 0}, {1: 1})
    assert qasm_co != co
    assert not qasm_co.jaqal_programs
    assert not qasm_co.jaqal_program


@pytest.mark.skipif(
    not importlib.util.find_spec("qiskit_superstaq"),
    reason="Skipping test as `qiskit_superstaq` is not installed.",
)
def test_read_json_pulse_gate_circuits() -> None:
    qss = pytest.importorskip("qiskit_superstaq", reason="qiskit-superstaq is not installed")
    import qiskit  # noqa: PLC0415

    circuit = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg meas[2];
        h q[0];
        cx q[0],q[1];
        barrier q[0],q[1];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        """
    )

    qc_pulse = qiskit.QuantumCircuit(2)
    qc_pulse.h(0)
    qc_pulse.cx(0, 1)

    json_dict = {
        "qasm_strs": json.dumps([circuit]),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc_pulse),
        "pulse_start_times": [[0, 10]],
    }

    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit

    pulse_output = out.pulse_gate_circuit
    assert pulse_output == qc_pulse
    assert hasattr(pulse_output, "op_start_times")
    assert pulse_output.op_start_times == [0, 10]

    json_dict = {
        "qasm_strs": json.dumps([circuit, circuit]),
        "initial_logical_to_physicals": "[[], []]",
        "final_logical_to_physicals": "[[], []]",
        "pulse_gate_circuits": qss.serialization.serialize_circuits([qc_pulse, qc_pulse]),
        "pulse_start_times": [[0, 10], [0, 100]],
    }
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]

    pulse_output = out.pulse_gate_circuits
    assert pulse_output == [qc_pulse, qc_pulse]
    assert all(hasattr(p_out, "op_start_times") for p_out in pulse_output)
    assert pulse_output[1].op_start_times == [0, 100]

    with (
        mock.patch.dict("sys.modules", {"qiskit_superstaq": None}),
        pytest.warns(UserWarning, match=r"qiskit-superstaq is required"),
    ):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_gate_circuit is None

    json_dict["pulse_gate_circuits"] = "not-a-serialized-circuit"
    with pytest.warns(
        UserWarning,
        match=r"Your compiled pulse gate circuits could not be deserialized.",
    ):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.pulse_gate_circuits is None


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json_aqt() -> None:
    importlib.reload(gss.compiler_output)

    circuit = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg meas[4];
        h q[0];
        h q[1];
        h q[2];
        h q[3];
        barrier q[0],q[1],q[2],q[3];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        measure q[3] -> meas[3];
        """
    )
    state_str = gss.serialization.serialize({})
    initial_logical_to_physical = {i: i for i in range(4)}
    final_logical_to_physical = {i: 3 - i for i in range(4)}

    json_dict = {
        "qasm_strs": json.dumps([circuit]),
        "state_jp": state_str,
        "initial_logical_to_physicals": json.dumps([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
    }

    with pytest.warns(UserWarning, match=r"deserialize compiled pulse sequences"):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)

    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert not hasattr(out, "circuits")
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")

    with pytest.warns(UserWarning, match=r"deserialize compiled pulse sequences"):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical]
    assert out.initial_logical_to_physicals == [initial_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    with pytest.warns(UserWarning, match=r"deserialize compiled pulse sequences"):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)

    assert out.circuit == circuit
    assert out.seq is None

    # multiple circuits
    json_dict = {
        "qasm_strs": json.dumps([circuit, circuit]),
        "state_jp": state_str,
        "initial_logical_to_physicals": json.dumps(2 * [list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps(2 * [list(final_logical_to_physical.items())]),
    }

    with pytest.warns(UserWarning, match=r"deserialize compiled pulse sequences"):
        out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)

    assert out.circuits == [circuit, circuit]
    assert out.initial_logical_to_physicals == [
        initial_logical_to_physical,
        initial_logical_to_physical,
    ]
    assert out.final_logical_to_physicals == [final_logical_to_physical, final_logical_to_physical]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    # no sequence returned
    json_dict.pop("state_jp")
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.seq is None


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)
    circuit = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg meas[4];
        h q[0];
        h q[1];
        h q[2];
        h q[3];
        barrier q[0],q[1],q[2],q[3];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        measure q[3] -> meas[3];
        """
    )
    initial_logical_to_physical = {i: i for i in range(4)}
    final_logical_to_physical = {i: 3 - i for i in range(4)}
    state_str = gss.serialization.serialize(seq.__getstate__())
    json_dict = {
        "qasm_strs": json.dumps([circuit]),
        "state_jp": state_str,
        "initial_logical_to_physicals": json.dumps([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
    }

    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out.seq, "_readout")
    assert not hasattr(out, "circuits")

    # Serialized readout attribute for `aqt_zurich_qpu`:
    json_dict["readout_jp"] = state_str
    json_dict["readout_qubits"] = "[4, 5, 6, 7]"
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 1
    assert pickle.dumps(out.seq._readout) == pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuits")

    # Multiple circuits:
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert not hasattr(out, "circuit")

    json_dict = {
        "qasm_strs": json.dumps([circuit, circuit]),
        "state_jp": state_str,
        "readout_jp": state_str,
        "readout_qubits": "[4, 5, 6, 7]",
        "initial_logical_to_physicals": json.dumps(2 * [list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps(2 * [list(final_logical_to_physical.items())]),
    }
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert pickle.dumps(out.seq) == pickle.dumps(seq)
    assert isinstance(out.seq, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout, qtrl.sequencer.Sequence)
    assert isinstance(out.seq._readout._readout, qtrl.sequence_utils.readout._ReadoutInfo)
    assert out.seq._readout._readout.sequence is out.seq._readout
    assert out.seq._readout._readout.qubits == [4, 5, 6, 7]
    assert out.seq._readout._readout.n_readouts == 2
    assert not hasattr(out, "circuit")


def test_read_json_qscout() -> None:
    circuit = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg meas[1];
        h q[0];
        barrier q[0];
        measure q[0] -> meas[0];
        """
    )
    initial_logical_to_physical = {0: 0}
    final_logical_to_physical = {0: 13}

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

    json_dict = {
        "qasm_strs": json.dumps([circuit]),
        "initial_logical_to_physicals": json.dumps([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program],
    }

    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.initial_logical_to_physical == initial_logical_to_physical
    assert out.final_logical_to_physical == final_logical_to_physical
    assert out.jaqal_program == jaqal_program
    assert out.jaqal_programs == [jaqal_program]
    assert not hasattr(out, "initial_logical_to_physicals")
    assert not hasattr(out, "final_logical_to_physicals")

    json_dict = {
        "qasm_strs": json.dumps([circuit, circuit]),
        "initial_logical_to_physicals": json.dumps(2 * [list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps(2 * [list(final_logical_to_physical.items())]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = gss.compiler_output.CompilerOutput.read_json(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical, final_logical_to_physical]
    assert out.initial_logical_to_physicals == [
        initial_logical_to_physical,
        initial_logical_to_physical,
    ]
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")
    assert out.jaqal_programs == [jaqal_program, jaqal_program]
    assert out.jaqal_program == jaqal_program_as_subcircuits

    out = gss.compiler_output.CompilerOutput.read_json(
        json_dict, circuits_is_list=True, num_eca_circuits=1
    )
    assert out.circuits == [[circuit], [circuit]]
    assert out.initial_logical_to_physicals == [
        [initial_logical_to_physical],
        [initial_logical_to_physical],
    ]
    assert out.final_logical_to_physicals == [
        [final_logical_to_physical],
        [final_logical_to_physical],
    ]
    assert out.jaqal_programs == [jaqal_program, jaqal_program]

    out = gss.compiler_output.CompilerOutput.read_json(
        json_dict, circuits_is_list=False, num_eca_circuits=2
    )
    assert out.circuits == [circuit, circuit]
    assert out.final_logical_to_physicals == [final_logical_to_physical, final_logical_to_physical]
    assert out.initial_logical_to_physicals == [
        initial_logical_to_physical,
        initial_logical_to_physical,
    ]
    assert out.jaqal_programs == [jaqal_program_as_subcircuits]
