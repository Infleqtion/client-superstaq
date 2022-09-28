import importlib
import textwrap
from unittest import mock

import cirq
import general_superstaq as gss
import pytest

import cirq_superstaq as css


def test_compiler_output_repr() -> None:
    circuit = cirq.Circuit()
    assert (
        repr(css.compiler_output.CompilerOutput(circuit))
        == f"CompilerOutput({circuit!r}, None, None, None, None)"
    )

    circuits = [circuit, circuit]
    assert (
        repr(css.compiler_output.CompilerOutput(circuits))
        == f"CompilerOutput({circuits!r}, None, None, None, None)"
    )


def test_read_json_ibmq() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "pulses": gss.converters.serialize([mock.DEFAULT]),
    }

    out = css.compiler_output.read_json_ibmq(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.pulse_sequence == mock.DEFAULT
    assert not hasattr(out, "circuits")
    assert not hasattr(out, "pulse_sequences")

    out = css.compiler_output.read_json_ibmq(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert out.pulse_sequences == [mock.DEFAULT]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "pulse_sequence")

    with mock.patch.dict("sys.modules", {"qiskit": None}), pytest.warns(
        UserWarning, match="requires Qiskit Terra"
    ):
        out = css.compiler_output.read_json_ibmq(json_dict, circuits_is_list=False)
        assert out.circuit == circuit
        assert out.pulse_sequence is None

    with mock.patch("qiskit.__version__", "0.17.2"), pytest.warns(
        UserWarning, match="you have 0.17.2"
    ):
        out = css.compiler_output.read_json_ibmq(json_dict, circuits_is_list=True)
        assert out.circuits == [circuit]
        assert out.pulse_sequences is None


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json_aqt() -> None:
    importlib.reload(css.compiler_output)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = gss.converters.serialize({})
    pulse_lists_str = gss.converters.serialize([[[]]])

    json_dict: dict

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }

    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert not hasattr(out, "circuits")

    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit]
    assert not hasattr(out, "circuit")

    # multiple circuits
    pulse_lists_str = gss.converters.serialize([[[]], [[]]])
    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "state_jp": state_str,
        "pulse_lists_jp": pulse_lists_str,
    }
    out = css.compiler_output.read_json_aqt(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert not hasattr(out, "circuit")


def test_read_json_qscout() -> None:
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
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
        "jaqal_programs": [jaqal_program],
    }

    out = css.compiler_output.read_json_qscout(json_dict, circuits_is_list=False)
    assert out.circuit == circuit
    assert out.jaqal_program == jaqal_program
    assert not hasattr(out, "jaqal_programs")

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = css.compiler_output.read_json_qscout(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
    assert out.jaqal_programs == json_dict["jaqal_programs"]
    assert not hasattr(out, "jaqal_program")


def test_read_json_only_circuits() -> None:
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits(circuit),
    }

    out = css.compiler_output.read_json_only_circuits(json_dict, circuits_is_list=False)
    assert out.circuit == circuit

    json_dict = {
        "cirq_circuits": css.serialization.serialize_circuits([circuit, circuit]),
    }
    out = css.compiler_output.read_json_only_circuits(json_dict, circuits_is_list=True)
    assert out.circuits == [circuit, circuit]
