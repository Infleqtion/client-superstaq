import codecs
import importlib
import pickle
from unittest import mock

import cirq
import pytest

from cirq_superstaq import aqt


@mock.patch.dict("sys.modules", {"qtrl": None})
def test_read_json() -> None:
    importlib.reload(aqt)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = codecs.encode(pickle.dumps({}), "base64").decode()
    pulse_list_str = codecs.encode(pickle.dumps([]), "base64").decode()

    json_dict = {
        "compiled_circuit": cirq.to_json(circuit),
        "state_jp": state_str,
        "pulse_list_jp": pulse_list_str,
    }
    compiler_output = aqt.read_json(json_dict)

    assert compiler_output == aqt.AQTCompilerOutput(circuit)


def test_read_json_with_qtrl() -> None:  # pragma: no cover, b/c test requires qtrl installation
    qtrl = pytest.importorskip("qtrl", reason="qtrl not installed")
    seq = qtrl.sequencer.Sequence(n_elements=1)

    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(4)))
    state_str = codecs.encode(pickle.dumps(seq.__getstate__()), "base64").decode()
    pulse_list_str = codecs.encode(pickle.dumps([]), "base64").decode()

    json_dict = {
        "compiled_circuit": cirq.to_json(circuit),
        "state_jp": state_str,
        "pulse_list_jp": pulse_list_str,
    }
    compiler_output = aqt.read_json(json_dict)

    assert compiler_output.circuit == circuit
    assert pickle.dumps(compiler_output.seq) == pickle.dumps(seq)
