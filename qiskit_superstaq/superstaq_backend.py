# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
from typing import Any, Iterable, List, Union

import cirq
import cirq.contrib.qasm_import
import ply
import qiskit
import qiskit_superstaq as qss
import requests


class BarrierGateStatement(cirq.contrib.qasm_import._parser.QasmGateStatement):
    def __init__(self) -> None:
        pass

    def on(
        self, params: List[float], args: List[List[cirq.ops.Qid]], lineno: int
    ) -> Iterable[cirq.ops.Operation]:
        qubits = [q[0] for q in args]
        yield cirq.ops.IdentityGate(len(qubits)).on(*qubits).with_tags("barrier")


class SuperstaQBackend(qiskit.providers.BackendV1):
    def __init__(
        self, provider: "qss.superstaq_provider.SuperstaQProvider", url: str, backend: str
    ) -> None:
        self.url = url
        self._provider = provider
        self.configuration_dict = {
            "backend_name": backend,
            "backend_version": "n/a",
            "n_qubits": -1,
            "basis_gates": None,
            "gates": [],
            "local": False,
            "simulator": False,
            "conditional": False,
            "open_pulse": False,
            "memory": False,
            "max_shots": -1,
            "coupling_map": None,
        }
        super().__init__(
            configuration=qiskit.providers.models.BackendConfiguration.from_dict(
                self.configuration_dict
            ),
            provider=provider,
        )

    @classmethod
    def _default_options(cls) -> qiskit.providers.Options:
        return qiskit.providers.Options(shots=1000)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, qss.superstaq_backend.SuperstaQBackend):
            return False

        return (
            self._provider == other._provider
            and self.configuration_dict == other.configuration_dict
        )

    def qiskit_to_circuit_json(
        self,
        circuit: qiskit.QuantumCircuit,
    ) -> str:
        """Return a json payload string (based on cirq.to_json) for the given Qiskit circuit."""

        def p_quantum_arg_bit_line(self, p: ply.yacc.YaccProduction) -> None:  # type: ignore
            """qarg : ID '[' NATURAL_NUMBER ']' """
            reg = p[1]
            idx = p[3]
            arg_name = self.make_name(idx, reg)
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = cirq.LineQubit(idx)
            p[0] = [self.qubits[arg_name]]

        setattr(
            cirq.contrib.qasm_import._parser.QasmParser, "p_quantum_arg_bit", p_quantum_arg_bit_line
        )

        cirq.contrib.qasm_import._parser.QasmParser.all_gates["barrier"] = BarrierGateStatement()
        qasm = circuit.qasm()
        circuit = cirq.contrib.qasm_import._parser.QasmParser().parse(qasm).circuit
        return json.loads(cirq.to_json(circuit))

    def run(
        self, circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]], **kwargs: int
    ) -> "qss.superstaq_job.SuperstaQJob":

        if isinstance(circuits, qiskit.QuantumCircuit):
            circuits = [circuits]

        superstaq_json = {
            "circuits": [self.qiskit_to_circuit_json(circuit) for circuit in circuits],
            "backend": self.name(),
            "shots": kwargs.get("shots"),
            "ibmq_token": kwargs.get("ibmq_token"),
            "ibmq_hub": kwargs.get("ibmq_hub"),
            "ibmq_group": kwargs.get("ibmq_group"),
            "ibmq_project": kwargs.get("ibmq_project"),
            "ibmq_pulse": kwargs.get("ibmq_pulse"),
        }

        headers = {
            "Authorization": self._provider.get_access_token(),
            "Content-Type": "application/json",
        }

        res = requests.post(
            self.url + "/" + qss.API_VERSION + "/multi_job",
            json=superstaq_json,
            headers=headers,
            verify=(self.url == qss.API_URL),
        )

        res.raise_for_status()
        response = res.json()
        if "ids" not in response:
            raise Exception

        #  we make a virtual job_id that aggregates all of the individual jobs
        # into a single one, that comma-separates the individual jobs:
        job_id = ",".join(response["ids"])
        job = qss.superstaq_job.SuperstaQJob(self, job_id)

        return job
