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

import os
from typing import List, Union

import qiskit
import requests

import qiskit_superstaq as qss


class SuperstaQProvider(qiskit.providers.ProviderV1):
    """Provider for SuperstaQ backend.

    Typical usage is:

    .. code-block:: python

        import qiskit_superstaq as qss

        ss_provider = qss.superstaq_provider.SuperstaQProvider('MY_TOKEN')

        backend = ss_provider.get_backend('my_backend')

    where `'MY_TOKEN'` is the access token provided by SuperstaQ,
    and 'my_backend' is the name of the desired backend.

    Attributes:
        access_token (str): The access token.
        name (str): Name of the provider instance.
        url (str): The url that the API is hosted on.
    """

    def __init__(
        self,
        access_token: str,
        url: str = os.getenv("SUPERSTAQ_REMOTE_HOST") or qss.API_URL,
    ) -> None:
        self.access_token = access_token
        self._name = "superstaq_provider"
        self.url = url

    def __str__(self) -> str:
        return f"<SuperstaQProvider(name={self._name})>"

    def __repr__(self) -> str:
        repr1 = f"<SuperstaQProvider(name={self._name}, "
        return repr1 + f"access_token={self.access_token})>"

    def get_backend(self, backend: str) -> "qss.superstaq_backend.SuperstaQBackend":
        return qss.superstaq_backend.SuperstaQBackend(provider=self, url=self.url, backend=backend)

    def get_access_token(self) -> str:
        return self.access_token

    def backends(self) -> List[qss.superstaq_backend.SuperstaQBackend]:
        # needs to be fixed (#469)
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
                qss.superstaq_backend.SuperstaQBackend(provider=self, url=self.url, backend=name)
            )

        return backends

    def aqt_compile(
        self, circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]]
    ) -> "qss.aqt.AQTCompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        if isinstance(circuits, qiskit.QuantumCircuit):
            json_dict = {"qasm_strs": [circuits.qasm()]}
            circuits_list = False
        else:
            json_dict = {"qasm_strs": [c.qasm() for c in circuits]}
            circuits_list = True

        headers = {
            "Authorization": self.get_access_token(),
            "Content-Type": "application/json",
        }
        res = requests.post(
            self.url + "/" + qss.API_VERSION + "/aqt_compile",
            json=json_dict,
            headers=headers,
            verify=(self.url == qss.API_URL),
        )
        res.raise_for_status()
        json_dict = res.json()

        from qiskit_superstaq import aqt

        return aqt.read_json(json_dict, circuits_list)
