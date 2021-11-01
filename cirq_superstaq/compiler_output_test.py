import importlib
import textwrap
from unittest import mock

import applications_superstaq
import cirq

import cirq_superstaq
from cirq_superstaq import compiler_output


def test_aqt_out_repr() -> None:
    circuit = cirq.Circuit()
    assert (
        repr(compiler_output.CompilerOutput(circuit))
        == f"CompilerOutput({circuit!r}, None, None, None)"
    )

    circuits = [circuit, circuit]
    assert (
        repr(compiler_output.CompilerOutput(circuits))
        == f"CompilerOutput({circuits!r}, None, None, None)"
    )


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(compiler_output)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = applications_superstaq.converters.serialize({})
    pulse_lists_str = applications_superstaq.converters.serialize([[[]]])

    json_dict: dict

    json_dict = {
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = compiler_output.read_json_aqt(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    out = compiler_output.read_json_aqt(json_dict, circuits_list=True)
    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    # multiple circuits
    pulse_lists_str = applications_superstaq.converters.serialize([[[]], [[]]])
    json_dict = {
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = compiler_output.read_json_aqt(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")


def test_read_json_with_qscout() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    jaqal_program = textwrap.dedent(
        """\
                register allqubits[1]
                prepare_all
                R allqubits[0] -1.5707963267948966 1.5707963267948966
                Rz allqubits[0] -3.141592653589793
                measure_all
                """
    )

    json_dict = {
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits(circuit),
        "jaqal_programs": [jaqal_program],
    }

    out = compiler_output.read_json_qscout(json_dict, circuits_list=False)
    assert out.circuit == circuit
    assert out.jaqal_programs == jaqal_program

    json_dict = {
        "cirq_circuits": cirq_superstaq.serialization.serialize_circuits([circuit, circuit]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = compiler_output.read_json_qscout(json_dict, circuits_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.jaqal_programs == json_dict["jaqal_programs"]
