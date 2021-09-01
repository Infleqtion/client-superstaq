import codecs
import importlib
import pickle
from unittest import mock

import cirq
import pytest

from cirq_superstaq import aqt


def test_aqt_compiler_output_repr() -> None:
    circuit = cirq.Circuit()
    assert repr(aqt.AQTCompilerOutput(circuit)) == f"AQTCompilerOutput({circuit!r}, None)"

    circuits = [circuit, circuit]
    assert repr(aqt.AQTCompilerOutput(circuits)) == f"AQTCompilerOutput({circuits!r}, None)"


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(aqt)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = codecs.encode(pickle.dumps({}), "base64").decode()

    json_dict: dict

    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit)],
        "state_jp": state_str,
    }

    compiler_output = aqt.read_json(json_dict, circuits_list=False)
    assert compiler_output.circuit == circuit
    assert not hasattr(compiler_output, "circuits")

    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit]
    assert not hasattr(compiler_output, "circuit")

    # multiple circuits
    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit), cirq.to_json(circuit)],
        "state_jp": state_str,
    }
    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit, circuit]
    assert not hasattr(compiler_output, "circuit")


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = codecs.encode(pickle.dumps(seq.__getstate__()), "base64").decode()

    json_dict: dict

    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit)],
        "state_jp": state_str,
    }

    compiler_output = aqt.read_json(json_dict, circuits_list=False)
    assert compiler_output.circuit == circuit
    assert not hasattr(compiler_output, "circuits")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)

    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit]
    assert not hasattr(compiler_output, "circuit")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)

    # multiple circuits
    json_dict = {
        "cirq_circuits": [cirq.to_json(circuit), cirq.to_json(circuit)],
        "state_jp": state_str,
    }
    compiler_output = aqt.read_json(json_dict, circuits_list=True)
    assert compiler_output.circuits == [circuit, circuit]
    assert not hasattr(compiler_output, "circuit")
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)
