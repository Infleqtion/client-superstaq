import codecs
import pickle
from unittest.mock import MagicMock, patch

import qiskit

import qiskit_superstaq as qss


def test_provider() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    assert str(ss_provider.get_backend("ibmq_qasm_simulator")) == str(
        qss.superstaq_backend.SuperstaQBackend(
            provider=ss_provider,
            url=qss.API_URL,
            backend="ibmq_qasm_simulator",
        )
    )

    assert str(ss_provider) == "<SuperstaQProvider(name=superstaq_provider)>"

    assert (
        repr(ss_provider) == "<SuperstaQProvider(name=superstaq_provider, access_token=MY_TOKEN)>"
    )

    backend_names = [
        "aqt_device",
        "ionq_device",
        "rigetti_device",
        "ibmq_botoga",
        "ibmq_casablanca",
        "ibmq_jakarta",
        "ibmq_qasm_simulator",
    ]

    backends = []
    for name in backend_names:
        backends.append(
            qss.superstaq_backend.SuperstaQBackend(
                provider=ss_provider, url=qss.API_URL, backend=name
            )
        )

    assert ss_provider.backends() == backends


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    out_qasm_str = """OPENQASM 2.0;
    include "qelib1.inc";

    //Qubits: [4, 5]
    qreg q[2];


    cz q[0],q[1];
    """
    expected_qc = qiskit.QuantumCircuit(2)
    expected_qc.cz(0, 1)

    mock_post.return_value.json = lambda: {
        "qasm_strs": [out_qasm_str],
        "state_jp": codecs.encode(pickle.dumps({}), "base64").decode(),
    }
    compiler_output = provider.aqt_compile(qc)
    assert compiler_output.circuit == expected_qc
    assert not hasattr(compiler_output, "circuits")

    compiler_output = provider.aqt_compile([qc])
    assert compiler_output.circuits == [expected_qc]
    assert not hasattr(compiler_output, "circuit")

    mock_post.return_value.json = lambda: {
        "qasm_strs": [out_qasm_str, out_qasm_str],
        "state_jp": codecs.encode(pickle.dumps({}), "base64").decode(),
    }
    compiler_output = provider.aqt_compile([qc, qc])
    assert compiler_output.circuits == [expected_qc, expected_qc]
    assert not hasattr(compiler_output, "circuit")
