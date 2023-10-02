# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import json
import os
import textwrap
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock, patch

import general_superstaq as gss
import numpy as np
import pytest
import qiskit
from general_superstaq import ResourceEstimate

import qiskit_superstaq as qss

if TYPE_CHECKING:
    from qiskit_superstaq.conftest import MockSuperstaqProvider


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_provider(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    assert str(fake_superstaq_provider) == "<SuperstaqProvider mock_superstaq_provider>"

    assert (
        repr(fake_superstaq_provider)
        == "<SuperstaqProvider(api_key=MY_TOKEN, name=mock_superstaq_provider)>"
    )

    assert str(fake_superstaq_provider.backends()[0]) == "ibmq_qasm_simulator"


def test_provider_args() -> None:
    with pytest.raises(ValueError, match="must be either 'ibm_cloud' or 'ibm_quantum'"):
        ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN", ibmq_channel="foo")

    ss_provider = qss.SuperstaqProvider(
        api_key="MY_TOKEN", ibmq_channel="ibm_quantum", ibmq_instance="instance", ibmq_token="token"
    )
    assert ss_provider._client.client_kwargs == dict(
        ibmq_channel="ibm_quantum", ibmq_instance="instance", ibmq_token="token"
    )


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_get_balance() -> None:
    ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    ss_provider._client = mock_client

    assert ss_provider.get_balance() == "$12,345.68"
    assert ss_provider.get_balance(pretty_output=False) == 12345.6789


@patch("requests.post")
def test_aqt_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[1, 4]]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }
    out = fake_superstaq_provider.aqt_compile(qc)
    assert out.circuit == qc
    assert out.final_logical_to_physical == {1: 4}
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")

    out = fake_superstaq_provider.aqt_compile([qc], atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{1: 4}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]], [[]]]),
    }
    out = fake_superstaq_provider.aqt_compile([qc, qc], test_options="yes")
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


def test_invalid_target_aqt_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid AQT target."):
        provider.aqt_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.post")
def test_aqt_compile_eca(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "state_jp": gss.serialization.serialize({}),
        "pulse_lists_jp": gss.serialization.serialize([[[]]]),
    }

    out = fake_superstaq_provider.aqt_compile(qc, num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "pulse_list")
    assert not hasattr(out, "final_logical_to_physical")

    out = fake_superstaq_provider.aqt_compile([qc], num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [[qc]]
    assert out.final_logical_to_physicals == [[{}]]

    with pytest.warns(DeprecationWarning, match="has been deprecated"):
        deprecated_out = fake_superstaq_provider.aqt_compile_eca(
            [qc], num_equivalent_circuits=1, random_seed=1234, atol=1e-2
        )
        assert deprecated_out.circuits == out.circuits
        assert deprecated_out.final_logical_to_physicals == out.final_logical_to_physicals


@patch("requests.post")
def test_ibmq_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)
    final_logical_to_physical = {0: 4, 1: 5}
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
        "pulses": gss.serialization.serialize([mock.DEFAULT]),
    }

    assert fake_superstaq_provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes"
    ) == qss.compiler_output.CompilerOutput(
        qc, final_logical_to_physical, pulse_sequences=mock.DEFAULT
    )
    assert fake_superstaq_provider.ibmq_compile(
        [qiskit.QuantumCircuit()]
    ) == qss.compiler_output.CompilerOutput(
        [qc], [final_logical_to_physical], pulse_sequences=[mock.DEFAULT]
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
    }

    assert fake_superstaq_provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes"
    ) == qss.compiler_output.CompilerOutput(qc, final_logical_to_physical, pulse_sequences=None)
    assert fake_superstaq_provider.ibmq_compile(
        [qiskit.QuantumCircuit()]
    ) == qss.compiler_output.CompilerOutput([qc], [final_logical_to_physical], pulse_sequences=None)


def test_invalid_target_ibmq_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid IBMQ target."):
        provider.ibmq_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_resource_estimate(
    mock_resource_estimate: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    resource_estimate = ResourceEstimate(0, 1, 2)

    mock_resource_estimate.return_value = {
        "resource_estimates": [{"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2}]
    }

    assert (
        fake_superstaq_provider.resource_estimate(qiskit.QuantumCircuit(), "ibmq_qasm_simulator")
        == resource_estimate
    )


@patch(
    "general_superstaq.superstaq_client._SuperstaqClient.resource_estimate",
)
def test_resource_estimate_list(
    mock_resource_estimate: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    resource_estimates = [ResourceEstimate(0, 1, 2), ResourceEstimate(3, 4, 5)]

    mock_resource_estimate.return_value = {
        "resource_estimates": [
            {"num_single_qubit_gates": 0, "num_two_qubit_gates": 1, "depth": 2},
            {"num_single_qubit_gates": 3, "num_two_qubit_gates": 4, "depth": 5},
        ]
    }

    assert (
        fake_superstaq_provider.resource_estimate([qiskit.QuantumCircuit()], "ibmq_qasm_simulator")
        == resource_estimates
    )


@patch("requests.post")
def test_qscout_compile(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
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
    out = fake_superstaq_provider.qscout_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.final_logical_to_physical == {0: 13}

    out = fake_superstaq_provider.qscout_compile([qc])
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{0: 13}]

    qc2 = qiskit.QuantumCircuit(2)
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc2]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13), (1, 11)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = fake_superstaq_provider.qscout_compile([qc, qc2])
    assert out.circuits == [qc, qc2]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13, 1: 11}]
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": "xx",
        "num_qubits": 2,
    }

    with pytest.raises(ValueError, match="At least 2 qubits are required"):
        _ = fake_superstaq_provider.qscout_compile([qc, qc2], num_qubits=1)


def test_invalid_target_qscout_compile(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid Sandia target."):
        fake_superstaq_provider.qscout_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.post")
@pytest.mark.parametrize("mirror_swaps", (True, False))
def test_qscout_compile_swap_mirror(
    mock_post: MagicMock, mirror_swaps: bool, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(1)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [""],
    }

    _ = fake_superstaq_provider.qscout_compile(qc, mirror_swaps=mirror_swaps)
    mock_post.assert_called_once()
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": "xx",
        "num_qubits": 1,
    }

    _ = fake_superstaq_provider.qscout_compile(qc, mirror_swaps=mirror_swaps, num_qubits=3)
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": mirror_swaps,
        "base_entangling_gate": "xx",
        "num_qubits": 3,
    }


@patch("requests.post")
@pytest.mark.parametrize("base_entangling_gate", ("xx", "zz"))
def test_qscout_compile_change_entangler(
    mock_post: MagicMock, base_entangling_gate: str, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(2)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[]]",
        "jaqal_programs": [""],
    }

    _ = fake_superstaq_provider.qscout_compile(qc, base_entangling_gate=base_entangling_gate)
    mock_post.assert_called_once()
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": base_entangling_gate,
        "num_qubits": 2,
    }

    _ = fake_superstaq_provider.qscout_compile(
        qc, base_entangling_gate=base_entangling_gate, num_qubits=4
    )
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": base_entangling_gate,
        "num_qubits": 4,
    }


def test_qscout_compile_wrong_entangler(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit()

    with pytest.raises(ValueError):
        _ = fake_superstaq_provider.qscout_compile(qc, base_entangling_gate="yy")


@patch("requests.post")
def test_qscout_compile_error_rates(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    circuit = qiskit.QuantumCircuit(1)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "final_logical_to_physicals": "[[]]",
        "jaqal_programs": [""],
    }

    _ = fake_superstaq_provider.qscout_compile(
        circuit, error_rates={(0, 1): 0.3, (0, 2): 0.2, (1,): 0.1}
    )
    mock_post.assert_called_once()
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "base_entangling_gate": "xx",
        "mirror_swaps": False,
        "error_rates": [[[0, 1], 0.3], [[0, 2], 0.2], [[1], 0.1]],
        "num_qubits": 3,
    }


@patch("requests.post")
def test_cq_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "final_logical_to_physicals": "[[[3, 0]]]",
    }
    out = fake_superstaq_provider.cq_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.final_logical_to_physical == {3: 0}

    out = fake_superstaq_provider.cq_compile([qc])
    assert out.circuits == [qc]
    assert out.final_logical_to_physicals == [{3: 0}]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "final_logical_to_physicals": "[[], []]",
    }
    out = fake_superstaq_provider.cq_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert out.final_logical_to_physicals == [{}, {}]


def test_invalid_target_cq_compile(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    with pytest.raises(ValueError, match="'ss_example_qpu' is not a valid CQ target."):
        fake_superstaq_provider.cq_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaqClient.supercheq",
)
def test_supercheq(
    mock_supercheq: mock.MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    circuits = [qiskit.QuantumCircuit()]
    fidelities = np.array([1])
    mock_supercheq.return_value = {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuits),
        "final_logical_to_physicals": "[[]]",
        "fidelities": gss.serialization.serialize(fidelities),
    }
    assert fake_superstaq_provider.supercheq([[0]], 1, 1) == (circuits, fidelities)


@patch("requests.post")
def test_dfe(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    mock_post.return_value.json = lambda: ["id1", "id2"]
    assert fake_superstaq_provider.submit_dfe(
        rho_1=(qc, "ss_example_qpu"),
        rho_2=(qc, "ss_example_qpu"),
        num_random_bases=5,
        shots=100,
    ) == ["id1", "id2"]

    with pytest.raises(ValueError, match="should contain a single circuit"):
        fake_superstaq_provider.submit_dfe(
            rho_1=([qc, qc], "ss_example_qpu"),
            rho_2=(qc, "ss_example_qpu"),
            num_random_bases=5,
            shots=100,
        )

    mock_post.return_value.json = lambda: 1
    assert fake_superstaq_provider.process_dfe(["1", "2"]) == 1


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
