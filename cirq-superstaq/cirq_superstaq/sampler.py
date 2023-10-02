# Copyright 2021 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A `cirq.Sampler` implementation for the Superstaq API."""
from __future__ import annotations

from typing import List

import cirq

import cirq_superstaq as css


class Sampler(cirq.Sampler):
    """A sampler that works against the Superstaq API. Users should get a sampler from the `sampler`
    method on `css.Service`.

    Example:

    .. code-block:: python

        service = css.Service(
            "Insert superstaq token that you received from https://superstaq.infleqtion.com"
            )
        q0, q1 = cirq.LineQubit.range(2)
        sampler = service.sampler("ibmq_qasm_simulator")
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))
        print(sampler.sample(circuit, repetitions=5))

            q(0)
        0   0
        1   0
        2   0
        3   1
        4   1

    """

    def __init__(
        self,
        service: css.service.Service,
        target: str,
    ) -> None:
        """Constructs the sampler, accessed from the `sampler` method on `css.Service`.

        Args:
            service: The service used to create this sample.
            target: The backend on which to run the job.
        """
        self._service = service
        self._target = target

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> List[cirq.ResultDict]:
        """Runs a sweep for the given circuit.

        Note:
            This creates jobs for each of the sweeps in the given sweepable, and then
            blocks until all of jobs are complete.

        Args:
            program: The circuit to sample from.
            params: The parameters to run with program.
            repetitions: The number of times to sample. Defaults to 1.

        Returns:
            A list of Cirq results, one for each parameter resolver.
        """
        resolvers = [resolver for resolver in cirq.to_resolvers(params)]
        job = self._service.create_job(
            circuits=[cirq.resolve_parameters(program, resolver) for resolver in resolvers],
            repetitions=repetitions,
            target=self._target,
        )
        cirq_results = [
            css.service.counts_to_results(job.counts(i), program, resolver)
            for i, resolver in enumerate(resolvers)
        ]
        return cirq_results
