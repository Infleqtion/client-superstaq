# Copyright 2025 Infleqtion
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
from general_superstaq import ResourceEstimate, testing

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
    assert (
        fake_superstaq_provider.backends()[0].name == "aqt_keysight_qpu"
    )  # First backend alphabetically.


def test_provider_args() -> None:
    with pytest.raises(ValueError, match=r"must be either 'ibm_cloud' or 'ibm_quantum_platform'"):
        ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN", ibmq_channel="foo")

    ss_provider = qss.SuperstaqProvider(
        api_key="MY_TOKEN",
        ibmq_channel="ibm_quantum_platform",
        ibmq_instance="instance",
        ibmq_token="token",
    )
    assert ss_provider._client.client_kwargs == {
        "ibmq_channel": "ibm_quantum_platform",
        "ibmq_instance": "instance",
        "ibmq_token": "token",
    }
    assert isinstance(ss_provider._client, gss.superstaq_client._SuperstaqClient)

    ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN", api_version="v0.3.0")
    assert isinstance(ss_provider._client, gss.superstaq_client._SuperstaqClientV3)

    with pytest.raises(ValueError, match=r"`api_version` can only take value 'v0.2.0' or 'v0.3.0'"):
        ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN", api_version="v0.4.0")


@patch.dict(os.environ, {"SUPERSTAQ_API_KEY": ""})
def test_get_balance() -> None:
    ss_provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    ss_provider._client = mock_client

    assert ss_provider.get_balance() == "12,345.68 credits"
    assert ss_provider.get_balance(pretty_output=False) == 12345.6789


@patch("requests.Session.post")
def test_get_job(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 0], [1, 1])
    backend = fake_superstaq_provider.get_backend("ibmq_brisbane_qpu")

    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id"], "status": "ready"},
    ):
        job = backend.run(qc, method="dry-run", shots=100)

    mock_post.return_value.json = lambda: {
        "job_id": {
            "status": "ready",
            "target": "ibmq_brisbane_qpu",
        }
    }

    assert job == fake_superstaq_provider.get_job("job_id")

    # multi circuit job with a comma separated job_id
    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id1,job_id2"], "status": "ready"},
    ):
        job = backend.run([qc, qc], method="dry-run", shots=100)

    mock_post.return_value.json = lambda: {
        "job_id1": {
            "status": "ready",
            "target": "ibmq_brisbane_qpu",
        },
        "job_id2": {
            "status": "ready",
            "target": "ibmq_brisbane_qpu",
        },
    }
    assert job == fake_superstaq_provider.get_job("job_id1,job_id2")

    # job ids belonging to different targets
    with patch(
        "general_superstaq.superstaq_client._SuperstaqClient.create_job",
        return_value={"job_ids": ["job_id1,job_id2"], "status": "ready"},
    ):
        job = backend.run([qc, qc], method="dry-run", shots=100)

    mock_post.return_value.json = lambda: {
        "job_id1": {
            "status": "ready",
            "target": "ibmq_brisbane_qpu",
        },
        "job_id2": {
            "status": "ready",
            "target": "ibmq_fez_qpu",
        },
    }
    with pytest.raises(
        gss.SuperstaqException, match=r"Job ids belong to jobs at different targets"
    ):
        fake_superstaq_provider.get_job("job_id1,job_id2")


@patch("requests.Session.post")
def test_aqt_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[[0, 1]]]",
        "final_logical_to_physicals": "[[[1, 4]]]",
    }
    out = fake_superstaq_provider.aqt_compile(qc)
    assert out.circuit == qc
    assert out.initial_logical_to_physical == {0: 1}
    assert out.final_logical_to_physical == {1: 4}
    assert not hasattr(out, "circuits")

    out = fake_superstaq_provider.aqt_compile([qc], atol=1e-2)
    assert out.circuits == [qc]
    assert out.initial_logical_to_physicals == [{0: 1}]
    assert out.final_logical_to_physicals == [{1: 4}]
    assert not hasattr(out, "circuit")
    mock_post.assert_called_with(
        f"{fake_superstaq_provider._client.url}/aqt_compile",
        headers=fake_superstaq_provider._client.headers,
        verify=fake_superstaq_provider._client.verify_https,
        json={
            "qiskit_circuits": qss.serialize_circuits(qc),
            "target": "aqt_keysight_qpu",
            "options": json.dumps({"atol": 1e-2}),
        },
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "initial_logical_to_physicals": "[[], []]",
        "final_logical_to_physicals": "[[], []]",
    }
    out = fake_superstaq_provider.aqt_compile([qc, qc], test_options="yes")
    assert out.circuits == [qc, qc]
    assert out.initial_logical_to_physicals == [{}, {}]
    assert out.final_logical_to_physicals == [{}, {}]
    assert not hasattr(out, "circuit")


def test_invalid_target_aqt_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid AQT target."):
        provider.aqt_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.Session.post")
def test_aqt_compile_eca(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
    }

    out = fake_superstaq_provider.aqt_compile(qc, num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [qc]
    assert out.initial_logical_to_physicals == [{}]
    assert out.final_logical_to_physicals == [{}]
    assert not hasattr(out, "circuit")
    assert not hasattr(out, "initial_logical_to_physical")
    assert not hasattr(out, "final_logical_to_physical")

    out = fake_superstaq_provider.aqt_compile([qc], num_eca_circuits=1, random_seed=1234, atol=1e-2)
    assert out.circuits == [[qc]]
    assert out.initial_logical_to_physicals == [[{}]]
    assert out.final_logical_to_physicals == [[{}]]

    with pytest.warns(DeprecationWarning, match=r"has been deprecated"):
        deprecated_out = fake_superstaq_provider.aqt_compile_eca(
            [qc], num_equivalent_circuits=1, random_seed=1234, atol=1e-2
        )
    assert deprecated_out.circuits == out.circuits
    assert deprecated_out.initial_logical_to_physicals == out.initial_logical_to_physicals
    assert deprecated_out.final_logical_to_physicals == out.final_logical_to_physicals


@patch("requests.Session.post")
def test_ibmq_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(8)
    qc.cz(4, 5)
    initial_logical_to_physical = {0: 0, 1: 1}
    final_logical_to_physical = {0: 4, 1: 5}
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": json.dumps([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
        "pulse_gate_circuits": qss.serialization.serialize_circuits(qc),
    }

    assert fake_superstaq_provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes", target="ibmq_fake_qpu"
    ) == qss.compiler_output.CompilerOutput(
        qc, initial_logical_to_physical, final_logical_to_physical, pulse_gate_circuits=qc
    )
    assert fake_superstaq_provider.ibmq_compile(
        [qiskit.QuantumCircuit()], target="ibmq_fake_qpu"
    ) == qss.compiler_output.CompilerOutput(
        [qc],
        [initial_logical_to_physical],
        [final_logical_to_physical],
        pulse_gate_circuits=[qc],
    )

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": json.dumps([list(initial_logical_to_physical.items())]),
        "final_logical_to_physicals": json.dumps([list(final_logical_to_physical.items())]),
    }

    assert fake_superstaq_provider.ibmq_compile(
        qiskit.QuantumCircuit(), test_options="yes", target="ibmq_fake_qpu"
    ) == qss.compiler_output.CompilerOutput(
        qc, initial_logical_to_physical, final_logical_to_physical
    )
    assert fake_superstaq_provider.ibmq_compile(
        [qiskit.QuantumCircuit()], target="ibmq_fake_qpu"
    ) == qss.compiler_output.CompilerOutput(
        [qc], [initial_logical_to_physical], [final_logical_to_physical]
    )
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "dd_strategy": "adaptive",
        "dynamical_decoupling": True,
    }

    assert fake_superstaq_provider.ibmq_compile(
        qiskit.QuantumCircuit(), dd_strategy="standard", test_options="yes", target="ibmq_fake_qpu"
    ) == qss.compiler_output.CompilerOutput(
        qc, initial_logical_to_physical, final_logical_to_physical
    )
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "dd_strategy": "standard",
        "dynamical_decoupling": True,
        "test_options": "yes",
    }


def test_invalid_target_ibmq_compile() -> None:
    provider = qss.SuperstaqProvider(api_key="MY_TOKEN")
    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid IBMQ target."):
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
        fake_superstaq_provider.resource_estimate(qiskit.QuantumCircuit(), "ibmq_fake_qpu")
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
        fake_superstaq_provider.resource_estimate([qiskit.QuantumCircuit()], "ibmq_fake_qpu")
        == resource_estimates
    )


@patch("requests.Session.post")
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
        "initial_logical_to_physicals": json.dumps([[(0, 1)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)]]),
        "jaqal_programs": [jaqal_program],
    }
    out = fake_superstaq_provider.qscout_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.initial_logical_to_physical == {0: 1}
    assert out.final_logical_to_physical == {0: 13}

    out = fake_superstaq_provider.qscout_compile([qc])
    assert out.circuits == [qc]
    assert out.initial_logical_to_physicals == [{0: 1}]
    assert out.final_logical_to_physicals == [{0: 13}]

    qc2 = qiskit.QuantumCircuit(2)
    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc2]),
        "initial_logical_to_physicals": json.dumps([[(0, 1)], [(0, 1), (1, 2)]]),
        "final_logical_to_physicals": json.dumps([[(0, 13)], [(0, 13), (1, 11)]]),
        "jaqal_programs": [jaqal_program, jaqal_program],
    }
    out = fake_superstaq_provider.qscout_compile([qc, qc2])
    assert out.circuits == [qc, qc2]
    assert out.initial_logical_to_physicals == [{0: 1}, {0: 1, 1: 2}]
    assert out.final_logical_to_physicals == [{0: 13}, {0: 13, 1: 11}]
    assert json.loads(mock_post.call_args.kwargs["json"]["options"]) == {
        "mirror_swaps": False,
        "base_entangling_gate": "xx",
        "num_qubits": 2,
    }

    with pytest.raises(ValueError, match=r"At least 2 qubits are required"):
        _ = fake_superstaq_provider.qscout_compile([qc, qc2], num_qubits=1)


def test_invalid_target_qscout_compile(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid QSCOUT target."):
        fake_superstaq_provider.qscout_compile(qiskit.QuantumCircuit(), target="ss_example_qpu")


@patch("requests.Session.post")
@pytest.mark.parametrize("mirror_swaps", [True, False])
def test_qscout_compile_swap_mirror(
    mock_post: MagicMock, mirror_swaps: bool, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(1)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": json.dumps([[(0, 1)]]),
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


@patch("requests.Session.post")
@pytest.mark.parametrize("base_entangling_gate", ["xx", "zz"])
def test_qscout_compile_change_entangler(
    mock_post: MagicMock, base_entangling_gate: str, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    qc = qiskit.QuantumCircuit(2)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[]]",
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

    with pytest.raises(ValueError, match=r"`base_entangling_gate` must be"):
        _ = fake_superstaq_provider.qscout_compile(qc, base_entangling_gate="yy")


@patch("requests.Session.post")
def test_qscout_compile_error_rates(
    mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider
) -> None:
    circuit = qiskit.QuantumCircuit(1)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(circuit),
        "initial_logical_to_physicals": "[[]]",
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


@patch("requests.Session.post")
def test_cq_compile(mock_post: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits(qc),
        "initial_logical_to_physicals": "[[[0, 1]]]",
        "final_logical_to_physicals": "[[[3, 0]]]",
    }
    out = fake_superstaq_provider.cq_compile(qc, test_options="yes")
    assert out.circuit == qc
    assert out.initial_logical_to_physical == {0: 1}
    assert out.final_logical_to_physical == {3: 0}

    out = fake_superstaq_provider.cq_compile([qc])
    assert out.circuits == [qc]
    assert out.initial_logical_to_physicals == [{0: 1}]
    assert out.final_logical_to_physicals == [{3: 0}]

    mock_post.return_value.json = lambda: {
        "qiskit_circuits": qss.serialization.serialize_circuits([qc, qc]),
        "initial_logical_to_physicals": "[[], []]",
        "final_logical_to_physicals": "[[], []]",
    }
    out = fake_superstaq_provider.cq_compile([qc, qc])
    assert out.circuits == [qc, qc]
    assert out.initial_logical_to_physicals == [{}, {}]
    assert out.final_logical_to_physicals == [{}, {}]


def test_invalid_target_cq_compile(fake_superstaq_provider: MockSuperstaqProvider) -> None:
    with pytest.raises(ValueError, match=r"'ss_example_qpu' is not a valid CQ target."):
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
        "initial_logical_to_physicals": "[[]]",
        "final_logical_to_physicals": "[[]]",
        "fidelities": gss.serialization.serialize(fidelities),
    }
    assert fake_superstaq_provider.supercheq([[0]], 1, 1) == (circuits, fidelities)


@patch("requests.Session.post")
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

    with pytest.raises(TypeError, match=r"should contain a single `qiskit.QuantumCircuit`"):
        fake_superstaq_provider.submit_dfe(
            rho_1=([qc, qc], "ss_example_qpu"),
            rho_2=(qc, "ss_example_qpu"),
            num_random_bases=5,
            shots=100,
        )

    mock_post.return_value.json = lambda: 1
    assert fake_superstaq_provider.process_dfe(["1", "2"]) == 1


@patch("requests.Session.get")
def test_get_targets(mock_get: MagicMock, fake_superstaq_provider: MockSuperstaqProvider) -> None:
    mock_get.return_value.json = {"superstaq_targets": testing.TARGET_LIST}
    assert fake_superstaq_provider.get_targets() == testing.RETURNED_TARGETS
