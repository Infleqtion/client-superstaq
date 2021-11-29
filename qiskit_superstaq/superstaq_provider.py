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
from typing import List, Optional, Union

import applications_superstaq
import qiskit
from applications_superstaq import finance
from applications_superstaq import logistics
from applications_superstaq import superstaq_client
from applications_superstaq import user_config

import qiskit_superstaq as qss


class SuperstaQProvider(
    qiskit.providers.ProviderV1, finance.Finance, logistics.Logistics, user_config.UserConfig
):
    """Provider for SuperstaQ backend.

    Typical usage is:

    .. code-block:: python

        import qiskit_superstaq as qss

        ss_provider = qss.superstaq_provider.SuperstaQProvider('MY_TOKEN')

        backend = ss_provider.get_backend('my_backend')

    where `'MY_TOKEN'` is the access token provided by SuperstaQ,
    and 'my_backend' is the name of the desired backend.

    Args:
         Args:
            remote_host: The location of the API in the form of a URL. If this is None,
                then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
                If that variable is not set, then this uses
                `https://superstaq.super.tech/{api_version}`,
                where `{api_version}` is the `api_version` specified below.
            api_key: A string key which allows access to the API. If this is None,
                then this instance will use the environment variable  `SUPERSTAQ_API_KEY`. If that
                variable is not set, then this will raise an `EnvironmentError`.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the API.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.
        Raises:
            EnvironmentError: if the `api_key` is None and has no corresponding environment
                variable set.
    """

    def __init__(
        self,
        remote_host: Optional[str] = None,
        api_key: Optional[str] = None,
        default_target: str = None,
        api_version: str = applications_superstaq.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ) -> None:
        self._name = "superstaq_provider"
        self.remote_host = (
            remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or applications_superstaq.API_URL
        )
        self.api_key = api_key or os.getenv("SUPERSTAQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Parameter api_key was not specified and the environment variable "
                "SUPERSTAQ_API_KEY was also not set."
            )

        self._client = superstaq_client._SuperstaQClient(
            client_name="qiskit-superstaq",
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def __str__(self) -> str:
        return f"<SuperstaQProvider(name={self._name})>"

    def __repr__(self) -> str:
        repr1 = f"<SuperstaQProvider(name={self._name}, "
        return repr1 + f"api_key={self.api_key})>"

    def get_backend(self, backend: str) -> "qss.superstaq_backend.SuperstaQBackend":
        return qss.superstaq_backend.SuperstaQBackend(
            provider=self, remote_host=self.remote_host, backend=backend
        )

    def get_access_token(self) -> Optional[str]:
        return self.api_key

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
                qss.superstaq_backend.SuperstaQBackend(
                    provider=self, remote_host=self.remote_host, backend=name
                )
            )

        return backends

    def _http_headers(self) -> dict:
        return {
            "Authorization": self.get_access_token(),
            "Content-Type": "application/json",
            "X-Client-Name": "qiskit-superstaq",
            "X-Client-Version": qss.API_VERSION,
        }

    def aqt_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "keysight",
    ) -> "qss.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, qiskit.QuantumCircuit)

        json_dict = self._client.aqt_compile(
            {"qiskit_circuits": serialized_circuits, "backend": target}
        )

        from qiskit_superstaq import compiler_output

        return compiler_output.read_json_aqt(json_dict, circuits_list)

    def qscout_compile(
        self,
        circuits: Union[qiskit.QuantumCircuit, List[qiskit.QuantumCircuit]],
        target: str = "qscout",
    ) -> "qss.compiler_output.CompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: qiskit QuantumCircuit(s)
        Returns:
            object whose .circuit(s) attribute is an optimized qiskit QuantumCircuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized qiskit.QuantumCircuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        serialized_circuits = qss.serialization.serialize_circuits(circuits)
        circuits_list = not isinstance(circuits, qiskit.QuantumCircuit)
        json_dict = self._client.qscout_compile(
            {"qiskit_circuits": serialized_circuits, "backend": target}
        )

        from qiskit_superstaq import compiler_output

        return compiler_output.read_json_qscout(json_dict, circuits_list)
