import codecs
import importlib
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union

import cirq

try:
    import qtrl.sequencer
except ModuleNotFoundError:
    pass


@dataclass
class AQTCompilerOutput:
    circuit: cirq.Circuit
    seq: Optional["qtrl.sequencer.Sequence"] = None


@dataclass
class AQTCompilerOutputMulti:
    circuits: List[cirq.Circuit]
    seq: Optional["qtrl.sequencer.Sequence"] = None


def read_json(json_dict: dict) -> Union[AQTCompilerOutput, AQTCompilerOutputMulti]:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /aqt_{multi_}compile endpoint
    Returns:
        a AQTCompilerOutput object with the compiled circuit or a AQTCompilerOutputMulti object
        with a list of compiled circuits. If qtrl is available locally, the returned object
        also stores the pulse sequence in the .seq attribute.
    """

    seq = None
    if importlib.util.find_spec(
        "qtrl"
    ):  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs
        state_str = json_dict["state_jp"]
        state = pickle.loads(codecs.decode(state_str.encode(), "base64"))

        seq = qtrl.sequencer.Sequence(n_elements=1)
        seq.__setstate__(state)
        seq.compile()

    if "compiled_circuit" in json_dict:
        compiled_circuit = cirq.read_json(json_text=json_dict["compiled_circuit"])
        return AQTCompilerOutput(compiled_circuit, seq)

    compiled_circuits = [cirq.read_json(json_text=c) for c in json_dict["compiled_circuits"]]
    return AQTCompilerOutputMulti(compiled_circuits, seq)
