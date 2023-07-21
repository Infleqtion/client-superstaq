# pylint: disable=missing-function-docstring,missing-class-docstring
import general_superstaq as gss


def test_pretty_print_compiler_output() -> None:
    """Tests printing human-readable version of compiler output."""
    # Compiler output with three empty cirucits
    compiler_output_repr = (
        "CompilerOutput([cirq.Circuit(), cirq.Circuit(), cirq.Circuit()], [{}, {}, {}], [Schedule(,"
        " name='sched3'), Schedule(, name='sched4'), Schedule(, name='sched5')], None, None, None)"
    )
    circuit_reprs = ["cirq.Circuit()", "cirq.Circuit()", "cirq.Circuit()"]
    circuit_drawings = ["", "", ""]
    assert gss.pretty_printing.pretty_print_compiler_output(
        compiler_output_repr, circuit_reprs, circuit_drawings
    ) == (
        "CompilerOutput(\n"
        "    [\n"
        "        cirq.Circuit(),\n"
        "        \n"
        "        cirq.Circuit(),\n"
        "        \n"
        "        cirq.Circuit(),\n"
        "    ],\n"
        "    [{}, {}, {}],\n"
        '    [Schedule(, name="sched3"), Schedule(, name="sched4"), Schedule(, name="sched5")'
        "],\n"
        "    None,\n"
        "    None,\n"
        "    None,\n)"
        "\n"
    )

    # Compiler output with a single X-gate circuit
    compiler_output_repr = (
        "CompilerOutput([cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "])], [{cirq.LineQubit(0): cirq.LineQubit(0)}], [Schedule((0, Play(Drag(duration=160, "
        "sigma=40, beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched0\")], None, None, None)"
    )
    circuit_reprs = [
        "cirq.Circuit([\n    cirq.Moment(\n        cirq.X(cirq.LineQubit(0)),\n    ),\n])"
    ]
    circuit_drawings = ["0: ───X───"]
    assert gss.pretty_printing.pretty_print_compiler_output(
        compiler_output_repr, circuit_reprs, circuit_drawings
    ) == (
        "CompilerOutput(\n"
        "    [0: ───X───],\n"
        "    [{cirq.LineQubit(0): cirq.LineQubit(0)}],\n"
        "    [\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched0",\n'
        "        )\n"
        "    ],\n"
        "    None,\n"
        "    None,\n"
        "    None,\n"
        ")\n"
    )

    # Compiler output with a compiled Hadamard circuit
    compiler_output_repr = (
        "CompilerOutput(<qiskit.circuit.quantumcircuit.QuantumCircuit object at 0x7fd2cc458ca0>, "
        "{0: 0}, Schedule((0, ShiftPhase(-1.5707963268, DriveChannel(0))), "
        "(0, ShiftPhase(-1.5707963268, ControlChannel(1))), (0, Play(Drag(duration=160, sigma=40, "
        "beta=-1.053810766518306, amp=0.1987049754148708, angle=0.022620855474399307, "
        "name='X90p_d0'), DriveChannel(0), name='X90p_d0')), (160, ShiftPhase(-1.5707963268, "
        'DriveChannel(0))), (160, ShiftPhase(-1.5707963268, ControlChannel(1))), name="sched2"), '
        "None, None, None)"
    )
    circuit_reprs = ["<qiskit.circuit.quantumcircuit.QuantumCircuit object at 0x7fd2cc458ca0>"]
    circuit_drawings = [
        (
            "     ┌─────────┐┌─────────┐┌─────────┐\n"
            "q_0: ┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├\n"
            "     └─────────┘└─────────┘└─────────┘"
        )
    ]
    assert gss.pretty_printing.pretty_print_compiler_output(
        compiler_output_repr, circuit_reprs, circuit_drawings
    ) == (
        "CompilerOutput(\n"
        "         ┌─────────┐┌─────────┐┌─────────┐\n"
        "    q_0: ┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├\n"
        "         └─────────┘└─────────┘└─────────┘,\n"
        "    {0: 0},\n"
        "    Schedule(\n"
        "        (0, ShiftPhase(-1.5707963268, DriveChannel(0))),\n"
        "        (0, ShiftPhase(-1.5707963268, ControlChannel(1))),\n"
        "        (\n"
        "            0,\n"
        "            Play(\n"
        "                Drag(\n"
        "                    duration=160,\n"
        "                    sigma=40,\n"
        "                    beta=-1.053810766518306,\n"
        "                    amp=0.1987049754148708,\n"
        "                    angle=0.022620855474399307,\n"
        '                    name="X90p_d0",\n'
        "                ),\n"
        "                DriveChannel(0),\n"
        '                name="X90p_d0",\n'
        "            ),\n"
        "        ),\n"
        "        (160, ShiftPhase(-1.5707963268, DriveChannel(0))),\n"
        "        (160, ShiftPhase(-1.5707963268, ControlChannel(1))),\n"
        '        name="sched2",\n'
        "    ),\n"
        "    None,\n"
        "    None,\n"
        "    None,\n"
        ")\n"
    )

    # Compiler output with five X-gate circuits
    compiler_output_repr = (
        "CompilerOutput([cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "]), cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "]), cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "]), cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "]), cirq.Circuit([\n"
        "    cirq.Moment(\n"
        "        cirq.X(cirq.LineQubit(0)),\n"
        "    ),\n"
        "])], [{cirq.LineQubit(0): cirq.LineQubit(0)}, {cirq.LineQubit(0): cirq.LineQubit(0)}, "
        "{cirq.LineQubit(0): cirq.LineQubit(0)}, {cirq.LineQubit(0): cirq.LineQubit(0)}, "
        "{cirq.LineQubit(0): cirq.LineQubit(0)}], [Schedule((0, Play(Drag(duration=160, sigma=40, "
        "beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched1\"), Schedule((0, Play(Drag(duration=160, "
        "sigma=40, beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched2\"), Schedule((0, Play(Drag(duration=160, "
        "sigma=40, beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched3\"), Schedule((0, Play(Drag(duration=160, "
        "sigma=40, beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched4\"), Schedule((0, Play(Drag(duration=160, "
        "sigma=40, beta=-1.040499642039664, amp=0.4033317218845395, angle=0.0, name='Xp_d0'), "
        "DriveChannel(0), name='Xp_d0')), name=\"sched5\")], None, None, None)"
    )
    circuit_reprs = [
        "cirq.Circuit([\n    cirq.Moment(\n        cirq.X(cirq.LineQubit(0)),\n    ),\n])"
    ] * 5
    circuit_drawings = ["0: ───X───"] * 5
    assert gss.pretty_printing.pretty_print_compiler_output(
        compiler_output_repr, circuit_reprs, circuit_drawings
    ) == (
        "CompilerOutput(\n"
        "    [\n"
        "        0: ───X───,\n"
        "        \n"
        "        0: ───X───,\n"
        "        \n"
        "        0: ───X───,\n"
        "        \n"
        "        0: ───X───,\n"
        "        \n"
        "        0: ───X───,\n"
        "    ],\n"
        "    [\n"
        "        {cirq.LineQubit(0): cirq.LineQubit(0)},\n"
        "        {cirq.LineQubit(0): cirq.LineQubit(0)},\n"
        "        {cirq.LineQubit(0): cirq.LineQubit(0)},\n"
        "        {cirq.LineQubit(0): cirq.LineQubit(0)},\n"
        "        {cirq.LineQubit(0): cirq.LineQubit(0)},\n"
        "    ],\n"
        "    [\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched1",\n'
        "        ),\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched2",\n'
        "        ),\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched3",\n'
        "        ),\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched4",\n'
        "        ),\n"
        "        Schedule(\n"
        "            (\n"
        "                0,\n"
        "                Play(\n"
        "                    Drag(\n"
        "                        duration=160,\n"
        "                        sigma=40,\n"
        "                        beta=-1.040499642039664,\n"
        "                        amp=0.4033317218845395,\n"
        "                        angle=0.0,\n"
        '                        name="Xp_d0",\n'
        "                    ),\n"
        "                    DriveChannel(0),\n"
        '                    name="Xp_d0",\n'
        "                ),\n"
        "            ),\n"
        '            name="sched5",\n'
        "        ),\n"
        "    ],\n"
        "    None,\n"
        "    None,\n"
        "    None,\n"
        ")\n"
    )
