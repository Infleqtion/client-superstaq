# Copyright 2026 Infleqtion
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

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import cirq
import numpy as np
import pydantic

import cirq_superstaq as css

from .ops import JumpChannel

if TYPE_CHECKING:
    import numpy.typing as npt


class SqaleNoiseParams(pydantic.BaseModel):
    """A Pydantic model to store Hilbert space noise parameters."""

    gr_static_overrotation_rads: float = 0.0
    gr_relative_overrotation: float = 0.0

    rz_relative_overrotation: float = 0.0
    rz_transition_matrix: list[list[float]] | None = None

    cz_phase_error: float = 0.0
    cz_transition_matrix: list[list[float]] | None = None

    movement_phase_error: float = 0.0

    classifier_errors: tuple[float, float] = (0.0, 0.0)
    initial_state_probs: list[float] = [0.0, 1.0, 0.0, 0.0, 0.0]

    def scale_by(self, scale_factor: float) -> SqaleNoiseParams:
        """Scales appropriate noise parameters by a constant factor."""
        return SqaleNoiseParams(
            gr_static_overrotation_rads=self.gr_static_overrotation_rads,
            gr_relative_overrotation=self.gr_relative_overrotation,
            cz_transition_matrix=self.cz_transition_matrix,
            cz_phase_error=self.cz_phase_error * scale_factor,
            rz_transition_matrix=self.rz_transition_matrix,
            rz_relative_overrotation=self.rz_relative_overrotation * scale_factor,
            movement_phase_error=self.movement_phase_error * scale_factor,
            classifier_errors=self.classifier_errors,
            initial_state_probs=self.initial_state_probs,
        )


# Noise model parameters used in 2LQ paper:
DEFAULT_PARAMS = SqaleNoiseParams(
    gr_static_overrotation_rads=0.0345,
    rz_relative_overrotation=0.012,
    rz_transition_matrix=[
        [0.0, 2 * 0.00066 * 0.1571993, 0.0, 0.0, 0.0],
        [0.0, 2 * 0.00066 * 0.1692620, 0.0, 0.0, 0.0],
        [0.0, 2 * 0.00066 * 0.3972632, 0.0, 0.0, 0.0],
        [0.0, 2 * 0.00066 * 0.2762855, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    cz_transition_matrix=[
        [0.00001740, 0.00018500, 0.00000486, 0.00016541, 0.0],
        [0.00001853, 0.00019750, 0.00000461, 0.00017774, 0.0],
        [0.00003113, 0.00042026, 0.00004574, 0.00120999, 0.0],
        [0.00004212, 0.00059021, 0.00005294, 0.00190101, 0.0],
        [0.00000000, 0.00385371, 0.00000000, 0.00000000, 0.0],
    ],
    cz_phase_error=0.0035,
    classifier_errors=(0.004, 0.028),
    initial_state_probs=[0.0065, 0.96, 0.0065, 0.027, 0.0],
)


class SqaleLeakageModel(cirq.NoiseModel):
    """A noise model for simulating leakage in SQALE."""

    def __init__(
        self,
        cz_transition_matrix: npt.ArrayLike | None = None,
        cz_phase_error: float = 0.0,
        gr_static_overrotation_rads: float = 0.0,
        gr_relative_overrotation: float = 0.0,
        rz_transition_matrix: npt.ArrayLike | None = None,
        rz_relative_overrotation: float = 0.0,
        movement_phase_error: float = 0.0,
        classifier_errors: tuple[float, float] = (0, 0),
        initial_state_probs: npt.ArrayLike | None = None,
        dimension: int = 5,
    ) -> None:
        """Initializes a SqaleLeakageModel.

        Args:
            cz_transition_matrix: The transition matrix for the CZ gate.
            cz_phase_error: Additional chance of Z error following CZ gate transitions.
            gr_static_overrotation_rads: The static over-rotation for the GR gate (in radians).
            gr_relative_overrotation: The relative over-rotation for the GR gate.
            rz_transition_matrix: The transition matrix for the RZ gate.
            rz_relative_overrotation: The relative over-rotation for the RZ gate.
            movement_phase_error: Chance of a phase flip after qubit motion.
            classifier_errors: A tuple containing the |g> and |e> state classifier errors.
            initial_state_probs: A list of probabilities for the initial state of the qubits.
            dimension: The dimension of the Hilbert space for the simulation.
        """
        self._cz_error_channel: cirq.Gate | None = None
        self._rz_leak_channel: JumpChannel | None = None

        # CZ Gate Errors
        if cz_transition_matrix is not None and np.any(cz_transition_matrix):
            cz_leak_channel = JumpChannel(cz_transition_matrix)
            phase_transitions = np.diag([2 * cz_phase_error, 2 * cz_phase_error])
            self._cz_error_channel = cz_leak_channel.then(JumpChannel(phase_transitions))
        elif cz_phase_error:
            self._cz_error_channel = cirq.phase_flip(cz_phase_error)

        # GR Gate Errors
        self._gr_static_overrotation_rads = gr_static_overrotation_rads
        self._gr_relative_overrotation = gr_relative_overrotation

        # RZ Gate Errors
        if rz_transition_matrix is not None and np.any(rz_transition_matrix):
            self._rz_leak_channel = JumpChannel(rz_transition_matrix)
        self._rz_relative_overrotation = rz_relative_overrotation

        self._movement_phase_error = movement_phase_error

        if initial_state_probs is None:
            initial_state_probs = np.zeros(dimension)
            initial_state_probs[1] = 1.0
        self._initial_state_probs = np.asarray(initial_state_probs)

        self.classifier_errors = classifier_errors
        self.dimension = dimension

    def noisy_operation(self, operation: cirq.Operation) -> Iterator[cirq.Operation]:
        """Returns a noisy version of the given cirq operation.

        Args:
            operation: The operation given.

        Returns:
            An iterator of noisy operations.
        """
        gate = operation.gate
        if gate == cirq.CZ:
            yield operation

            if self._cz_error_channel:
                yield from self._cz_error_channel.on_each(*operation.qubits)
        elif isinstance(gate, (css.RGate, css.ParallelRGate)):
            # This is how the hardware canonicalizes rotation angles:
            theta = (gate.theta - np.pi) % (-2 * np.pi) + np.pi
            phi = gate.phi
            if theta < 0:
                theta = -theta
                phi += np.pi
            phi %= 2 * np.pi

            new_gate = css.ParallelRGate(theta, phi, cirq.num_qubits(operation))
            yield new_gate.on(*operation.qubits)

            # Add overrotation component as a second GR gate
            theta_over = theta * self._gr_relative_overrotation
            theta_over += self._gr_static_overrotation_rads

            if theta_over:
                over_gate = css.ParallelRGate(theta_over, phi, cirq.num_qubits(operation))
                yield over_gate.on(*operation.qubits)

        elif isinstance(gate, cirq.ZPowGate):
            yield operation

            # Add overrotation component
            yield cirq.pow(operation, self._rz_relative_overrotation)

            if self._rz_leak_channel:
                yield from self._rz_leak_channel.on_each(*operation.qubits)

        elif isinstance(gate, cirq.MeasurementGate):
            assert not gate.confusion_map

            meas_error_array = np.zeros((self.dimension, self.dimension))
            meas_error_array[0, 0:4:2] = 1 - self.classifier_errors[0]
            meas_error_array[0, 1:4:2] = self.classifier_errors[1]
            meas_error_array[1, 0:4:2] = self.classifier_errors[0]
            meas_error_array[1, 1:4:2] = 1 - self.classifier_errors[1]
            meas_error_array[2:3, 4:] = 1  # Project all loss into "2" state

            assert np.allclose(meas_error_array.sum(0), 1)

            classifier_error_channel = JumpChannel(meas_error_array)

            yield from classifier_error_channel.on_each(*operation.qubits)
            yield cirq.measure(*operation.qubits, key=cirq.measurement_key_obj(operation))
        elif isinstance(gate, cirq.QubitPermutationGate):
            yield operation

            channel = cirq.phase_flip(self._movement_phase_error)
            yield from channel.on_each(*operation.qubits)
        else:
            yield operation

    def noisy_moments(
        self, moments: Iterable[cirq.Moment], system_qubits: Sequence[cirq.Qid]
    ) -> list[cirq.Moment]:
        """Returns noisy moments for a circuit.

        This method adds initial state preparation noise and then applies the noisy
        operation to each operation in the circuit.

        Args:
            moments: The moments of the cirq circuit.
            system_qubits: The qubits in the system.

        Returns:
            A list of noisy moments.
        """
        initial_transition_matrix = np.zeros((self.dimension, self.dimension))
        for i, prob in enumerate(self._initial_state_probs[: self.dimension]):
            initial_transition_matrix[i, 0] = prob

        initial_jumps = cirq.Moment(JumpChannel(initial_transition_matrix).on_each(*system_qubits))

        # Circuits always start with a pi pulse to flip qubits into the zero state
        initial_flip = css.ParallelRGate(np.pi, 0.0, len(system_qubits)).on(*system_qubits)

        circuit = cirq.Circuit(initial_flip, *moments)
        return [initial_jumps, *circuit.map_operations(self.noisy_operation).moments]
