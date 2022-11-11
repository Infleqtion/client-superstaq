# Pulse Manipulator
"""
PulseManipulator: Quickly and easily manipulate Qiskit pulse schedules.
"""

import qiskit
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict, Optional
from collections import defaultdict
from copy import deepcopy

VERBOSE = False
printv = lambda text: print(text) if VERBOSE else None

class PulseManipulator:
    """
    This class is a pulse schedule wrapper that allows more precise manipulation.
    """

    def __init__(self, schedule=None, backend=None):
        """Constructs intial PulseManipulator using qiskit pulse schedule."""
        if not isinstance(schedule, qiskit.pulse.Schedule):
            raise ValueError("Input schedule must be of type qiskit.pulse.Schedule.")

        if not isinstance(backend, qiskit.providers.Backend):
            raise ValueError("Input backend must be of type qiskit.providers.Backend")

        self._init_schedule = deepcopy(schedule)
        self._schedule = deepcopy(schedule)
        self._backend = backend
        self._id_to_inst_map = None
        self._set_id_to_instruction_map()
        self._coinciding_channels = self._get_coinciding_channels_map()


    @property
    def duration(self):
        """Returns the duration for the instruction."""
        return self._schedule.duration


    @property
    def name(self, inst_id=None):
        """Returns the name for the instruction."""
        _, inst = self._extract_instruction(inst_id=inst_id)
        return inst.name


    def get_start_time(self, inst_id=None):
        """Returns the start time for the instruction."""
        start_time, _ = self._extract_instruction(inst_id=inst_id)
        return start_time


    def get_stop_time(self, inst_id=None):
        """Returns the stop time for the instruction."""
        _, instruction = self._extract_instruction(inst_id=inst_id)
        return self.get_start_time(inst_id=inst_id) + instruction.duration


    def draw(self, time_range=None):
        """Draws pulse schedule to screen, inheriting qiskit's functionality."""
        # TODO: include a "pretty" draw
        return self._schedule.draw(time_range=time_range)


    def append(self, instruction, inplace=False):
        """Appends instruction to the end of the pulse schedule."""
        return self.insert(start_time=self.duration, instruction=instruction, inplace=inplace)


    def insert(self, start_time=None, instruction=None, inplace=False):
        """Inserts instruction at requested start time, unless it overlaps with other instructions.

        If the full duration of the instruction will fit, defaults to qiskit's insert. However,
        unlike qiskit's version, this allows inserting between two directly adjacent instructions.
        """
        assert start_time >= 0, "You must provide a nonnegative start time."
        assert (
            self._backend is not None
        ), (
            "This method requires knowledge about which pulse channels coincide, so you need to "
            "provide a backend. Use PulseManipulator.set_backend."
        )
        instruction = self._extract_instruction(pulse_obj=instruction)
        if start_time == 0:
            schedule = self.shift(
                shift_amount=instruction.duration, channel=instruction.channel, inplace=False
            )._schedule.insert(start_time, instruction, inplace=inplace)
        elif start_time >= self.duration or instruction.channel not in self._schedule.channels:
            schedule = self._schedule.insert(start_time, instruction, inplace=inplace)
        else:
            assert (
                isinstance(instruction.channel, qiskit.pulse.DriveChannel)
                or isinstance(instruction.channel, qiskit.pulse.ControlChannel)
            ), (
                "Except for appending or inserting at t=0, only insertions on DriveChannel and "
                "ControlChannel are currently supported."
            )
            coinciding_channels = self._coinciding_channels[instruction.channel]
            (
                (left_start_time, left_instruction), (right_start_time, right_instruction)
            ) = self._find_insertion_point(start_time, instruction.channel, *coinciding_channels)
            assert(
                (left_start_time, left_instruction) != (right_start_time, right_instruction)
            ), (
                f"The requested start_time {start_time} overlaps with an existing instruction from "
                f"{left_start_time} to {left_start_time + left_instruction.duration}"
            )
            schedule = self._perform_insertion(
                start_time,
                instruction,
                (left_start_time, left_instruction),
                (right_start_time, right_instruction),
                instruction.channel,
                coinciding_channels,
            )
        return self._update(schedule=schedule, inplace=inplace)
        

    def replace(self, inst_id=None, instruction=None, flip_amp=False, inplace=False):
        """Replaces instruction A with instruction B, regardless of each instruction's duration.

        Unlike qiskit's version, this shifts other instructions in the channel as necessary to
        allow replacing an instruction with a larger one. For replacing an instruction with a
        smaller one, we allow two versions, in each of which the surrounding instructions will: 1)
        "tight"=True: shift so that the distances away from the new instruction is the same as the
        distance away from the old instruction. 2) "tight"=False: remain in their original positions
        in time.
        """
        # TODO: Include "flip_amp" parameter to flip amplitude of instruction indicated by ID.
        # TODO: reimplement coinciding_channels so that passing in a channel input gives *every*
        # channel (including itself) it coincides with
        assert (
            self._backend is not None
        ), (
            "This method requires knowledge about which pulse channels coincide, so you need to "
            "provide a backend. Use PulseManipulator.set_backend."
        )
        old_start_time, old_instruction = self._extract_instruction(inst_id=inst_id)
        new_instruction = self._extract_instruction(pulse_obj=instruction)
        if new_instruction.duration <= old_instruction.duration:
            # TODO: this is the "non-tight" version -- implement a "tight" version by splitting in
            # half, inserting on left end, shifting right end left-wards, and merging.
            coinciding_channels = self._coinciding_channels[old_instruction.channel]
            schedule = self._schedule.exclude(
                channels=coinciding_channels,
                time_ranges=[(old_start_time, old_start_time + old_instruction.duration)]
            ).insert(old_start_time, new_instruction)
            return self._update(schedule, inplace)
        else:
            pass


    def flip_amp(self, inst_id=None, inplace=False):
        """Flips amplitude of specified instruction pulse."""
        if inplace:
            # TODO: don't want to access private variable, so instead make a new pulse with amp
            # parameter set to -amp
            start_time, instruction = self._extract_instruction(inst_id=inst_id)
            instruction.pulse._amp *= -1
            instruction._name = instruction.name.replace(
                "m", "%"
            ).replace("p", "m").replace("%", "p")
            return self
        else:
            # TODO: update using _set_id_to_inst_map?
            # TODO: provide inplace functionality, return self._update(schedule, inplace)
            return self._update(
                schedule=deepcopy(self._schedule), inplace=inplace
            ).flip_amp(inst_id=inst_id, inplace=True)


    def shift(self, shift_amount=None, channel=None, inplace=True):
        """Shifts instructions on any or all channels.

        This defaults to qiskit's version with a single parameter. Otherwise, allows shifting
        individual instructions or shifting one or more channels individually.
        """
        # TODO: try an "after" keyword to allow shifting instructions after a certain point in time
        # or after / before a certain instruction indicated by unique instruction ID
        assert (
            self._backend is not None
        ), (
            "This method requires knowledge about which pulse channels coincide, so you need to "
            "provide a backend. Use PulseManipulator.set_backend."
        )
        try:
            float(shift_amount)
        except ValueError:
            raise ValueError(f"{shift_amount} is not a valid input for shift amount.")

        if channel is None:
            self._schedule.shift(shift_amount)
            return

        assert(
            isinstance(channel, qiskit.pulse.channels.PulseChannel)
        ), "You need to provide a valid pulse channel."
        coinciding_channels = self._coinciding_channels[channel]
        schedule = (
            self._schedule.filter(channels=[channel, *coinciding_channels]).shift(shift_amount)
            | self._schedule.exclude(channels=[channel, *coinciding_channels])
        )
        return self._update(schedule, inplace)


    def measure(self):
        """Adds measurement pulses.

        This also allows measuring individual qubits, as well as including or excluding the delays
        that are inserted by default to avoid ringdown for mid-circuit measurements.
        """
        # TODO: This should include the ability to perform mid-circuit measurement, taking a time
        # parameter for where to insert it.


    def remove_measure(self):
        """Removes measurement pulses.

        By default (w/ no parameters), removes all measurements. Otherwise, either removes (1)
        measurements at the end of the pulse schedule or (2) measurments at a particular time.
        """
        pass


    def get_parameters(self):
        """Prints instruction pulse parameters (e.g., amplitude, sigma, beta, width)."""
        pass


    def set_parameters(self):
        """Sets instruction pulse parameters.

        All instruction pulses will have "amp" and "sigma"; only Drag pulses have "beta"; only
        GaussianSquare pulses have "width".

        Args:
            parameters (dict): 

        E.g., pv.set_parameters({"amp": 1, "sigma": 40})
        """
        pass


    def reset(self, inplace=False):
        """Resets to the schedule used to initialize it."""
        return self._update(deepcopy(self._init_schedule), inplace)


    def _update(self, schedule=None, inplace=False):
        """Updates PulseManipulator schedule inplace or out-of-place."""
        if inplace:
            self._schedule = schedule
            self._set_id_to_instruction_map()
            return self
        else:
            return PulseManipulator(schedule=schedule, backend=self._backend)
        

    def copy(self):
        """Returns a new copy of class instance."""
        pass


    def set_backend(self, backend=None):
        """Sets qiskit backend object.

        This is necessary for insert method, which requires knowledge about which drive channel
        corresponds to a control channel.
        """
        assert isinstance(
            backend, qiskit.providers.Backend
        ), "Input backend must be of type qiskit.providers.Backend."
        self._backend = backend
        self._coinciding_channels_map = self._get_coinciding_channels_map()


    def get_instruction_pulse(self, inst_id: int = None) -> qiskit.pulse.ParametricPulse:
        """Gets pulse waveform for an instruction.

        E.g., this is particularly useful for instructions on a phase-shifted channel (i.e., those
        that follow a virtual Z), to see what the un-shifted pulse is.
        """
        _, instruction = self._extract_instruction(inst_id=inst_id)
        return instruction.pulse


    def get_instruction_samples(self, inst_id: int = None) -> np.ndarray:
        """Gets samples for instruction's pulse waveform."""
        _, instruction = self._extract_instruction(inst_id=inst_id)
        return instruction.pulse.get_waveform().samples


    def get_qiskit_schedule(self) -> qiskit.pulse.Schedule:
        """Returns a copy of underlying qiskit pulse schedule (without unique IDs)."""
        return deepcopy(self._schedule)


    def _set_id_to_instruction_map(
        self,
        schedule: qiskit.pulse.Schedule = None,
        with_label_indexing: bool = True,
        inplace: bool = True,
    ):
        """Sets map from unique instruction ID to instruction, for schedule manipulation methods."""
        if schedule is None:
            schedule = deepcopy(self._schedule)
        new_schedule = deepcopy(schedule)
        id_to_inst_map = {}
        for idx, (start_time, instruction) in enumerate(schedule.instructions):
            if not isinstance(
                instruction, qiskit.pulse.Play
            ) or isinstance(instruction.channel, qiskit.pulse.MeasureChannel):
                continue
            if with_label_indexing:
                new_instruction = self._extract_instruction(instruction, inst_id=idx)
            else:
                new_instruction = self._extract_instruction(instruction)
            assert(
                instruction.duration == new_instruction.duration
            ), "Instructions should only differ by waveform label."
            id_to_inst_map[idx] = (start_time, new_instruction)
            remainder = new_schedule.exclude(
                lambda inst: inst[0] == start_time and inst[1] == instruction
            )
            # TODO: replacing the instruction on filtered schedule should work...? But seems more
            # unweildy...
            new_schedule = (new_instruction << start_time) | remainder
        if inplace:
            self._schedule = new_schedule
            self._id_to_inst_map = id_to_inst_map
        else:
            return new_schedule


    def _get_waveform_label(self, instruction, new_inst_id=None, flip_amp=False):
        """Gets LaTeX name corresponding to instruction, with an optional unique ID."""
        # TODO: how should the instruction.name is None case be handled? No integer index is added.
        if waveform.name is None:
            return ""
        pulse_name, sign, channels = re.split("(m|p)+", instruction.name)
        if flip_amp:
            sign = sign.replace("m", "").replace("p", "-")
        else:
            sign = sign.replace("m", "-").replace("p", "")
        symbol, angle, _ = pulse_name.partition("90")
        angle = sign + "\pi" + ("/2" if angle == "90" else "")
        if len(channels.split("_")) > 2:
            latex = f"$\overline{{{symbol}}}({angle})$"
        else:
            latex = f"${symbol}({angle})$"
        if new_inst_id is None:
            return latex
        return f"{latex}\n{new_inst_id}"


    def _extract_instruction(
        self,
        pulse_obj=None,
        inst_id=None,
        channel=None,
        flip_amp=False,
    ):
        """Gets instruction from a qiskit pulse object.

        This allows users to pass in either (a) an instruction (e.g., qiskit.pulse.Play), (b) a
        pulse waveform (qiskit.pulse.Waveform), or (c) an array of samples for a pulse waveform
        """
        # TODO: handle an input schedule (perhaps containing multiple instructions)
        if inst_id is not None and (pulse_obj is None and channel is None and flip_amp is None):
            assert(
                self._id_to_inst_map.get(inst_id, None) is not None
            ), (
                "You need to provide a valid instruction ID."
                f"Select one of {list(self._id_to_inst_map.keys())}."
            )
            return self._id_to_inst_map[inst_id]

        if isinstance(pulse_obj, qiskit.pulse.Play):
            waveform_label = self._get_waveform_label(pulse_obj, inst_id=inst_id)
            if waveform_label is not None:
                return qiskit.pulse.Play(
                    type(pulse_obj.pulse)(**pulse_obj.pulse.parameters, name=waveform_label),
                    pulse_obj.channel,
                    pulse_obj.name,
                )
            return deepcopy(pulse_obj)

        if isinstance(pulse_obj, qiskit.pulse.Waveform):
            assert channel is not None, "You need to provide a channel for a pulse waveform."
            assert name is not None, "Provide a name for the pulse waveform."
            return qiskit.pulse.Play(
                pulse_obj,
                channel=channel,
                name=f"{pulse_obj.name}_instruction",
            )

        if isinstance(pulse_obj, np.array) and (pulse_obj.ndim == 1):
            assert channel is not None, "You need to provide a channel for a pulse waveform."
            assert name is not None, "Provide a name for the pulse waveform"
            waveform_label = self._get_waveform_label(pulse_obj, inst_id=inst_id)
            return qiskit.pulse.Play(
                qiskit.pulse.Waveform(samples=pulse_obj, name=waveform_label),
                channel=channel,
                name=f"{waveform_label}_instruction",
            )

        """ A slightly more compact way to extract instructions...
        if isinstance(pulse_obj, qiskit.pulse.Play):
            parameters = pulse_obj.pulse.parameters
            if flip_amp:
                parameters["amp"] *= -1
            waveform_label = self._get_waveform_label(pulse_obj, inst_id=inst_id, flip_amp=flip_amp)
            waveform = type(pulse_obj.pulse)(**parameters, name=waveform_label)
        elif isinstance(pulse_obj, qiskit.pulse.Waveform):
            assert channel is not None, "You need to provide a channel for a pulse waveform."
            assert name is not None, "Provide a name for the pulse waveform."
            waveform = pulse_obj
        elif isinstance(pulse_obj, np.array) and (pulse_obj.ndim == 1):
            assert channel is not None, "You need to provide a channel for a pulse waveform."
            assert name is not None, "Provide a name for the pulse waveform"
            waveform_label = self._get_waveform_label(pulse_obj, inst_id=inst_id, flip_amp=flip_amp)
            waveform = qiskit.pulse.Waveform(samples=pulse_obj, name=waveform_label),
        else:
            raise ValueError(f"Input format of type {type(inst)} unrecognized.")

        return qiskit.pulse.Play(
            waveform,
            channel=channel,
            name=f"{waveform_label}_instruction",
        )
        """



    # TODO: add this everywhere _coinciding_channels is used
    def _get_coinciding_channels(self, channel=None):
        """"""
        assert(
            isinstance(channel, qiskit.pulse.Channel)
        ), "You must provide a valid qiskit pulse channel."
        assert(
            channel in self._coinciding_channels
        ), f"Requested channel {channel} does not exist on this backend."
        return self._coinciding_channels[channel]


    def _get_coinciding_channels_map(self):
        """Get map from pulse channel to all channels that physically coincide with it (including
        itself).

        E.g., On most IBM Q devices, DriveChannel(0) and ControlChannel(0) map to the same phyysical
        channel.
        """
        if self._backend is None:
            return dict()
        coinciding_channels_map: Dict = defaultdict(set)
        for (
            (drive_qubit, _), (control_channel,)
        ) in self._backend.configuration().control_channels.items():
            drive_channel = qiskit.pulse.DriveChannel(drive_qubit)
            coinciding_channels_map[drive_channel].add(control_channel)
            coinciding_channels_map[control_channel].add(drive_channel)
        for channel, collisions in coinciding_channels_map.items():
            for collision in collisions:
                coinciding_channels_map[channel] = coinciding_channels_map[channel].union(
                    coinciding_channels_map[collision]
                )
        return coinciding_channels_map

    
    
    def _find_insertion_point(self, start_time, *channels):
        """Finds pair of instructions between which new instruction can be scheduled."""
        instructions_to_search = self._schedule.filter(channels=channels).instructions
        for idx in range(len(instructions_to_search) - 1):
            left_start, left_instruction = instructions_to_search[idx]
            right_start, right_instruction = instructions_to_search[idx + 1]
            if start_time >= left_start + left_instruction.duration and start_time <= right_start:
                return (left_start, left_instruction), (right_start, right_instruction)
            if start_time > left_start and start_time < left_start + left_instruction.duration:
                return (left_start, left_instruction), (left_start, left_instruction)
            if start_time > right_start and start_time < right_start + right_instruction.duration:
                return (right_start, right_instruction), (right_start, right_instruction)
    
    
    def _perform_insertion(
        self,
        start_time,
        instruction,
        left_scheduled_instruction,
        right_scheduled_instruction,
        *channels,
    ):
        """Inserts instruction between two others on a single channel."""
        left_start, left_instruction = left_scheduled_instruction
        right_start, right_instruction = right_scheduled_instruction
        left_half = self._schedule.filter(
            channels=channels, time_ranges=[(0, left_start + left_instruction.duration)]
        )
        right_half = self._schedule.filter(
            channels=channels,
            time_ranges=[(right_start + right_instruction.duration, self.duration)],
        )
        remainder = self._schedule.exclude(channels=channels)
        left_half.insert(start_time, instruction, inplace=True)
        shift_amount = max(0, left_start + left_half.duration - right_start)
        right_half.shift(shift_amount, inplace=True)
        return left_half | right_half | remainder

####################################################################################################

    def flip_amplitude(self, inst_id=None, inplace=True):
        """Flips amplitude of specified instruction pulse."""
        #TODO: make a stretch_pulse method following this design
        assert (
            self._id_to_scheduled_inst_map.get(instruction_id) is not None
        ), (
            f"You need to supply a valid instruction ID. Choose one of "
            f"{list(self._id_to_scheduled_inst_map.keys())}."
        )
        old_scheduled_instruction = self._id_to_inst_map.get(inst_id)
        start_time, old_instruction = self.get_instruction_and_start_time(inst_id=inst_id)
        new_instruction = self._get_new_instruction(old_instruction, flip_sign=True)
        new_scheduled_instruction = self._schedule_instruction(
            new_instruction,
            start_time,
            instruction_id=instruction_id,
        )
        printv("=" * 80)
        printv(f"Start time: {start_time}")
        printv(f"old instruction: {old_instruction.name}")
        printv(f"new instruction: {new_scheduled_instruction.instructions[0][1].name}")
        filtered_copy = self._schedule.filter(
            lambda inst: not isinstance(inst[1], qiskit.pulse.ShiftPhase),
            channels=[old_instruction.channel],
            time_ranges=[(start_time, start_time + old_instruction.duration)]
        )
        replaced_copy = filtered_copy.replace(
            old_instruction,
            new_scheduled_instruction.instructions[0][1],
        )
        remainder = self._schedule.exclude(
            lambda inst: inst[0] == start_time and inst[1] == old_instruction
        )
        schedule = replaced_copy | remainder
        if inplace:
            self._schedule = schedule
            self._id_to_scheduled_inst_map[instruction_id] = new_scheduled_instruction
            return self
        else:
            return PulseVisualization(schedule, backend=self._backend)


    ############################################### TODO: Perhaps sort these
    def _get_coinciding_channels(self):
        """This should return the input channel if there are no coinciding channels set. This makes
        it so that users can fully use PM without providing a backend (however as soon as a backend
        is provided, perhaps there should be a warning of some kind and an option to reorganize the
        pulse schedule.
        """
        pass

    def _schedule_instruction(self):
        pass


    def get_pulse_gates_circuit(self):
        pass


    def get_native_gates(self):
        # TODO: how to implement this? list of strs, and another function to get native inst given
        # an input str? or some other way -- pass in str name instead of instruction (I like this
        # one)
        pass


    def reset_qubit(self):
        # TODO: add a reset pulse, and maybe rename reset method to something else so I can call
        # this reset...
        pass
