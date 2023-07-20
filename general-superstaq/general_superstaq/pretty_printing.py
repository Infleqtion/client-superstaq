import typing

import black
import cirq
import qiskit

MAX_LINE_LENGTH = 120

def get_circuit_drawing_lines(circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit]):
    """Gets a list of the lines making up a circuit drawing."""
    if isinstance(circuit, qiskit.QuantumCircuit):
        drawing = str(circuit.draw(idle_wires=False))
    elif isinstance(circuit, cirq.Circuit):
        drawing = str(circuit)
    if drawing == "":
        return [repr(circuit)]
    return drawing.split("\n")


def get_circuit_placeholder(
    circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit], circuit_index: int = 0
):
    """Gets placeholder text for a given circuit in the compiler output that allows Black to
    correctly format it.

    We use Black to wrap lines and indent accordingly with respect to a maximum MAX_LINE_LENGTH.
    However, Black is not sufficient because (a) it doesn't treat the lines of a circuit drawing as
    a contiguous piece, and (b) it doesn't recognize the custom characters in some circuit drawings
    (e.g., Qiskit), so we instead replace each line with a unique placeholder (using only
    Black-recognizable characters) that we then replace *after* using Black to format the compiler
    output repr.

    We comma-separate the placeholder lines so that Black formatting
    converts, e.g.,

        # Assume the length of this line exceeds MAX_LINE_LENGTH:
        LINE_PLACEHOLDER_0, LINE_PLACEHOLDER_1, LINE_PLACEHOLDER_2,

    into

        LINE_PLACEHOLDER_0,
        LINE_PLACEHOLDER_1,
        LINE_PLACEHOLDER_2,

    which we then replace to get, e.g.,

        0: ───Rz(1.5π)───Rx(0.5π)───AceCR+-(Z side)───
                                    │
        1: ───Rx(0.5π)──────────────AceCR+-(X side)───,

    and so on for any given circuit drawing.
    """
    # Befor for loop, if circuit is empty or any given line is too long, use a single placeholder
    # for the entire circuit, to be replaced with the circuit repr.
    circuit_drawing_lines = get_circuit_drawing_lines(circuit)
    if (
        circuit_drawing_lines == [repr(circuit)]
        or len(max(circuit_drawing_lines, key=len)) > MAX_LINE_LENGTH
    ):
        return f"quantum_circuit_{circuit_index}"

    ################################################################################################
    # We create placeholders according to the following rules:
    #
    #     - (1) For a given circuit indicated by unique label M, every Nth line is given the
    #     placeholder "quantum_circuit_M_line_N".
    #
    #     - (2) Add padding (using "x" character) to the placeholder making it the same length as
    #     the original line so that Black formats it correctly.
    #
    #     - (3) If any given line is too long (greater than a fixed maximum length, e.g., 100), the
    #     entire set of lines is given the placeholder "quantum_circuit_X", to be replaced by the
    #     circuit repr (which Black can't parse on its own).
    #
    # ##############################################################################################
    placeholder_lines = []
    # TODO: try replacing the for loop with the commented-out line below, regarding parsing of
    # custom characters (can we get Black to deal with the error by itself?)
    # placeholder_lines = circuit_drawing_lines  # gives `KeyError` for qiskit
    for line_index, line in enumerate(circuit_drawing_lines):
        placeholder = f"quantum_circuit_{circuit_index}_{line_index}"
        if len(placeholder) < len(line):
            padding = "x" * (len(line) - len(placeholder))
            placeholder += padding
        placeholder_lines.append(placeholder)
    return ", ".join(placeholder_lines)


def replace_circuit_placeholder(text, circuit, circuit_index: int = 0):
    """Replace given circuit drawing placeholder w/ corresponding circuit drawing.

    Given the placeholder text for a circuit drawing, we consider each line and replace it with the
    corresponding line in the circuit drawing. In order to preserve Black's formatting, we need to
    replace the placeholder text line by line.
    """
    circuit_drawing_lines = get_circuit_drawing_lines(circuit)
    circuit_placeholder_lines = get_circuit_placeholder(circuit, circuit_index).split(", ")

    # This will remove commas at the end of each circuit placeholder line except for the last line,
    # in order to preserve comma-separation between circuits (and at the end of the last circuit, a
    # la comma separation for Google style docstrings:
    #
    #     LINE_PLACEHOLDER_0,
    #     LINE_PLACEHOLDER_1,
    #     LINE_PLACEHOLDER_2,
    #
    # becomes
    #
    #     CIRCUIT_DRAWING_LINE_0
    #     CIRCUIT_DRAWING_LINE_1
    #     CIRCUIT_DRAWING_LINE_2,
    #
    # Note that only one comma remains.
    for idx, (placeholder, line) in enumerate(
        zip(circuit_placeholder_lines, circuit_drawing_lines)
    ):
        if idx < len(circuit_placeholder_lines) - 1:
            text = text.replace(placeholder + ",", line)
        else:
            text = text.replace(placeholder, line)
    return text

def preprocess_compiler_output(raw_text, circuit_reprs, circuit_repr_placeholders):
    """Replace parts of the compiler output unrecognizable to Black with placeholders."""
    # First, replace components that Black cannot parse: (a) an empty schedule will contain a
    # parenthesis followed by a comma, (b) an empty logical-to-physical qubit mapping shows up as
    # enclosed braces.
    preprocessed_out = raw_text.replace("Schedule(,", "Schedule(None,").replace("{}", "empty_map")
    for circuit_repr, placeholder in zip(circuit_reprs, circuit_repr_placeholders):
        preprocessed_out = preprocessed_out.replace(circuit_repr, placeholder)
    return preprocessed_out


# Black prettifier; the goal is to get this to work on compiler output
def prettify(text, out):
    return (black.format_str(text, mode=black.Mode(line_length=MAX_LINE_LENGTH)), out)


def postprocess_compiler_output(pretty_out, out):
    """Replace each circuit drawing placeholder with the corresponding circuit drawing line."""
    # First, restore preprocessed non-circuit placeholders for (a) an empty schedule will contain
    # a parenthesis followed by a comma, (b) an empty logical-to-physical qubit mapping shows up as
    # enclosed braces.
    postprocessed_out = (
        pretty_out.replace("empty_map", "{}")
        .replace("Schedule(None,", "Schedule(,")
        .replace("newline_placeholder,", "")
    )
    if not out.has_multiple_circuits():
        return replace_circuit_placeholder(postprocessed_out, out.circuit)

    for circuit_index, circuit in enumerate(out.circuits):
        postprocessed_out = replace_circuit_placeholder(postprocessed_out, circuit, circuit_index)
    return postprocessed_out


def pretty_print(out):
    """Prints human-readable version of compiler output."""
    raw_text = repr(out)
    if not out.has_multiple_circuits():
        circuit_reprs = [repr(out.circuit)]
        circuit_repr_placeholders = [get_circuit_placeholder(out.circuit)]
    else:
        # This makes sure to insert newlines in between circuits for readability, by replacing each 
        # circuit repr (followed by a comma) with two lines: (1) the circuit placeholder, followed
        # by a commma, and (2) the newline placeholder, followed by a comma (so that Black, which
        # expects comma-separated input, wraps lines correctly). (The last circuit does not have a
        # comma after it in the compiler output repr, so we handle that by looking for just the
        # circuit repr and swapping it out with the corresponding circuit placeholder, leaving the
        # comma.)
        circuit_reprs = [repr(circuit) + "," for circuit in out.circuits]
        circuit_repr_placeholders = [
            get_circuit_placeholder(circuit, circuit_index) + ", newline_placeholder,"
            for circuit_index, circuit in enumerate(out.circuits)
        ]

        circuit_reprs[-1] = circuit_reprs[-1].rstrip(",")
        circuit_repr_placeholders[-1] = circuit_repr_placeholders[-1].rstrip(
            ", newline_placeholder,"
        )

    preprocessed_out = preprocess_compiler_output(
        raw_text, circuit_reprs, circuit_repr_placeholders
    )
    pretty_out, out = prettify(preprocessed_out, out)
    return postprocess_compiler_output(pretty_out, out)
