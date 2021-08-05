import codecs
import pickle
from typing import Union

import cirq


def read_json(json_dict: dict) -> Union[cirq.Circuit, tuple]:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Endpoint returns (JSON-serialized) a cirq circuit in the "compiled_circuit" key
    and a serialization of the qtrl sequence in "state_jp".

    We only deserialize the qtrl sequence if qtrl is available locally.
    """
    out_circuit = cirq.read_json(json_text=json_dict["compiled_circuit"])

    try:
        import qtrl
    except ModuleNotFoundError:
        return out_circuit

    if True:  # pragma: no cover b/c qtrl is not yet open source
        state_str = json_dict["state_jp"]
        state = pickle.loads(codecs.decode(state_str.encode(), "base64"))
        seq = qtrl.sequencer.Sequence(n_elements=1)
        seq.set_state(state)
        seq.compile()
        return (out_circuit, seq)
