# pylint: disable=missing-function-docstring,missing-class-docstring
import json
import os
import textwrap
from unittest import mock
from unittest.mock import MagicMock, patch

import general_superstaq as gss
import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate

import qiskit_superstaq as qss


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_provider() -> None:
    ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    assert str(ss_provider.get_backend("ibmq_qasm_simulator")) == str(
        qss.SuperstaqBackend(provider=ss_provider, target="ibmq_qasm_simulator")
    )

    assert str(ss_provider) == "<SuperstaqProvider superstaq_provider>"

    assert repr(ss_provider) == "<SuperstaqProvider(api_key=MY_TOKEN, name=superstaq_provider)>"

    targets = {
        "superstaq_targets": {
            "compile-and-run": [
                "ibmq_qasm_simulator",
                "ibmq_armonk_qpu",
                "ibmq_santiago_qpu",
                "ibmq_bogota_qpu",
                "ibmq_lima_qpu",
                "ibmq_belem_qpu",
                "ibmq_quito_qpu",
                "ibmq_statevector_simulator",
                "ibmq_mps_simulator",
                "ibmq_extended-stabilizer_simulator",
                "ibmq_stabilizer_simulator",
                "ibmq_manila_qpu",
                "aws_dm1_simulator",
                "aws_tn1_simulator",
                "ionq_ion_qpu",
                "aws_sv1_simulator",
                "rigetti_aspen-9_qpu",
            ],
            "compile-only": ["aqt_keysight_qpu", "sandia_qscout_qpu"],
        }
    }

    expected_backends = []
    for target in targets["superstaq_targets"]["compile-and-run"]:
        expected_backends.append(qss.SuperstaqBackend(provider=ss_provider, target=target))

    mock_client = MagicMock()
    mock_client.get_targets.return_value = targets
    ss_provider._client = mock_client
    assert ss_provider.backends() == expected_backends


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_get_balance() -> None:
    ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    ss_provider._client = mock_client

    assert ss_provider.get_balance() == "$12,345.68"
    assert ss_provider.get_balance(pretty_output=False) == 12345.6789


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[1, 4]]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }
    out = provider.aqt_compile(qc)
    assert out.circuit == qc
    assert out.final_logical_to_physical == {1: 4}
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = provider.aqt_compile([qc], atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{1: 4}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]], [[]]]),
    }
    out = provider.aqt_compile([qc, qc], test_options="yes")
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


def test_invalid_target_aqt_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid AQT target."):
        provider.aqt_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.post")
def test_aqt_compile_eca(mock_post: MagicMock) -> None:
    provider = qss.superstaq_provider.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }

    out = provider.aqt_compile(qc, num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "pulse_list")
    assert not hasattr(out, "final_logical_to_physical")

    out = provider.aqt_compile([qc], num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [[qc]]
    assert out.final_logical_to_physicals == [[{}]]

    with pytest.warns(DeprecationWarning, match="has been deprecated"):
        deprecated_out = provider.aqt_compile_eca(
            [qc], num_equivalent_circuits=1, random_seed=1234, atol=1e-2
        )
        assert deprecated_out.circuits == out.circuits
        assert deprecated_out.final_logical_to_physicals == out.final_logical_to_physicals


@patch("requests.post")
def test_ibmq_compile(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)
    final_logical_to_physical = {0: 4, 1: 5}
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
        "pulses": gss.serialization.serialize([mock.DEFAULT]),
    }

    assert provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes"
    ) == qss.compiler_output.CompilerOutput(
        qc, final_logical_to_physical, pulse_sequences=mock.DEFAULT
    )
    assert provider.ibmq_compile([qiskit.QuantumCircuit()]) == qss.compiler_output.CompilerOutput(
        [qc], [final_logical_to_physical], pulse_sequences=[mock.DEFAULT]
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
    }

    assert provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes"
    ) == qss.compiler_output.CompilerOutput(qc, final_logical_to_physical, pulse_sequences=None)
    assert provider.ibmq_compile([qiskit.QuantumCircuit()]) == qss.compiler_output.CompilerOutput(
        [qc], [final_logical_to_physical], pulse_sequences=None
    )


def test_invalid_target_ibmq_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid IBMQ target."):
        provider.ibmq_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_resource_estimate(mock_resource_estimate: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    resource_estimate = ResourceEstimate(0, 1, 2)

    mock_resource_estimate.return_value = {
        "resource_estimates": [{"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2}]
    }

    assert (
        provider.resource_estimate(qiskit.QuantumCircuit(), "ibmq_qasm_simulator")
        == resource_estimate
    )


@patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_resource_estimate_list(mock_resource_estimate: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    resource_estimates = [ResourceEstimate(0, 1, 2), ResourceEstimate(3, 4, 5)]

    mock_resource_estimate.return_value = {
        "resource_estimates": [
            {"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2},
            {"num_single_qubit_gates": 3, "num_two_qubit_gates": 4, "depth": 5},
        ]
    }

    assert (
        provider.resource_estimate([qiskit.QuantumCircuit()], "ibmq_qasm_simulator")
        == resource_estimates
    )


@patch("requests.post")
def test_qscout_compile(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)

    jaqal_program = textwrap.dedent(
        """\
        register allqubits[1]

        prepare_all
        R allqubits[0] -1.5707963267948966 1.5707963267948966
        Rz allqubits[0] -3.141592653589793
        measure_all
        """
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [jaqal_program],
    }
    out = provider.qscout_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.final_logical_to_physical == {0: 13}

    out = provider.qscout_compile([qc])
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{0: 13}]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = provider.qscout_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13}]


def test_invalid_target_qscout_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid Sandia target."):
        provider.qscout_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.post")
@pytest.mark.parametrize("mirror_swaps", (True, False))
def test_qscout_compile_swap_mirror(mock_post: MagicMock, mirror_swaps: bool) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit()

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [""],
    }
    _ = provider.qscout_compile(qc, mirror_swaps=mirror_swaps)
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert json.loads(kwargs["json"]["options"]) == {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": "xx",
    }


@patch("requests.post")
@pytest.mark.parametrize("base_entangling_gate", ("xx", "zz"))
def test_qscout_compile_change_entangler(mock_post: MagicMock, base_entangling_gate: str) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit()

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "jaqal_programs": [""],
    }
    _ = provider.qscout_compile(qc, base_entangling_gate=base_entangling_gate)
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert json.loads(kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": base_entangling_gate,
    }


@patch("requests.post")
def test_qscout_compile_wrong_entangler(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit()

    with pytest.raises(ValueError):
        _ = provider.qscout_compile(qc, base_entangling_gate="yy")


@patch("requests.post")
def test_cq_compile(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[3, 0]]]",
    }
    out = provider.cq_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.final_logical_to_physical == {3: 0}

    out = provider.cq_compile([qc])
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{3: 0}]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
    }
    out = provider.cq_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]


def test_invalid_target_cq_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid CQ target."):
        provider.cq_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.supercheq",
)
def test_supercheq(mock_supercheq: mock.MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="key")
    circuits = [qiskit.QuantumCircuit()]
    fidelities = np.array([1])
    mock_supercheq.return_value = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuits),
        "final_logical_to_physicals": "[[]]",
        "fidelities": gss.serialization.serialize(fidelities),
    }
    assert provider.supercheq([[0]], 1, 1) == (circuits, fidelities)


@patch("requests.post")
def test_target_info(mock_post: MagicMock) -> None:
    provider = qss.SuperstaqProvider(api_key="key")
    fake_data = {"target_info": {"backend_name": "ss_example_qpu", "max_experiments": 1234}}
    mock_post.return_value.json = lambda: fake_data
    assert provider.target_info("ss_example_qpu") == fake_data["target_info"]


def test_get_targets() -> None:
    provider = qss.SuperstaqProvider(api_key="key", remote_host="http://example.com")
    mock_client = mock.MagicMock()
    targets = {
        "superstaq_targets": {
            "compile-and-run": [
                "ibmq_qasm_simulator",
                "ibmq_armonk_qpu",
                "ibmq_santiago_qpu",
                "ibmq_bogota_qpu",
                "ibmq_lima_qpu",
                "ibmq_belem_qpu",
                "ibmq_quito_qpu",
                "ibmq_statevector_simulator",
                "ibmq_mps_simulator",
                "ibmq_extended-stabilizer_simulator",
                "ibmq_stabilizer_simulator",
                "ibmq_manila_qpu",
                "aws_dm1_simulator",
                "aws_sv1_simulator",
                "d-wave_advantage-system4.1_qpu",
                "d-wave_dw-2000q-6_qpu",
                "aws_tn1_simulator",
                "rigetti_aspen-9_qpu",
                "d-wave_advantage-system1.1_qpu",
                "ionq_ion_qpu",
            ],
            "compile-only": ["aqt_keysight_qpu", "aqt_zurich_qpu", "sandia_qscout_qpu"],
        }
    }
    mock_client.get_targets.return_value = targets
    provider._client = mock_client

    assert provider.get_targets() == targets["superstaq_targets"]
