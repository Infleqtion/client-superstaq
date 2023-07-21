from typing import List

import black

MAX_LINE_LENGTH = 100


def get_circuit_drawing_lines(circuit_drawing: str, circuit_repr: str) -> List[str]:
    """Gets a list of the lines making up a circuit drawing.

    Args:
        circuit_drawing: The single-line str representation of a multi-line circuit drawing.
        circuit_repr: The repr for the circuit.

    Returns:
        The list of newline-separated circuit drawing lines.
    """
    if circuit_drawing == "":
        return [circuit_repr]
    return circuit_drawing.split("\n")


def get_circuit_placeholder(circuit_repr: str, circuit_drawing: str, circuit_index: int = 0) -> str:
    """Gets placeholder text for a given circuit in the compiler output for Black parsing.

    We want Black to format each circuit drawing (or repr if the drawing is too large). Black will
    wrap lines and indent accordingly with respect to a maximum MAX_LINE_LENGTH. However, Black is
    not sufficient because (a) it doesn't treat the lines of a circuit drawing as a contiguous
    piece, and (b) it doesn't recognize the custom characters in some circuit drawings, e.g., in
    Qiskit, so we instead replace each line with a unique placeholder (using only Black-recognizable
    characters) that we then replace *after* using Black to format the compiler output repr.

    We comma-separate the placeholder lines so that Black formatting converts, e.g.,

        # Assume this line exceeds MAX_LINE_LENGTH
        LINE_PLACEHOLDER_0, LINE_PLACEHOLDER_1, LINE_PLACEHOLDER_2,

    into

        LINE_PLACEHOLDER_0,
        LINE_PLACEHOLDER_1,
        LINE_PLACEHOLDER_2,

    which we then replace to get, e.g.,

        0: ───Rz(1.5π)───Rx(0.5π)───AceCR+-(Z side)───
                                    │
        1: ───Rx(0.5π)──────────────AceCR+-(X side)───,

    and so on for any given circuit drawing. We use "quantum_circuit_M_line_N" as the Nth line
    placeholder for the Mth quantum circuit in a sequence, or "quantum_circuit_M" as a placeholder
    for the repr if the circuit drawing is too large, i.e., any given line is greater than some
    fixed maximum length. We also add padding (using "x") to each placeholder to make it the same
    length as the original line so Black formats it correctly.

    Args:
        circuit_repr: The repr for the circuit.
        circuit_drawing: The single-line str representation of a multi-line circuit drawing.
        circuit_index: The index of the circuit if it is in a list.

    Returns:
        A string placeholder for the circuit repr or drawing that can be parsed by Black.
    """
    # Befor for loop, if circuit is empty or any given line is too long, use a single placeholder
    # for the entire circuit, to be replaced with the circuit repr.
    circuit_drawing_lines = get_circuit_drawing_lines(circuit_drawing, circuit_repr)
    if (
        circuit_drawing_lines == [circuit_repr]
        or len(max(circuit_drawing_lines, key=len)) > MAX_LINE_LENGTH
    ):
        placeholder = f"quantum_circuit_{circuit_index}"
        padding = "x" * (len(circuit_repr) - len(placeholder))
        return placeholder + padding

    placeholder_lines = []
    # TODO: try replacing the for loop with the commented-out line below, regarding parsing of
    # custom characters placeholder_lines = circuit_drawing_lines  # gives `KeyError` for qiskit;
    # can we get Black to deal with it?
    for line_index, line in enumerate(circuit_drawing_lines):
        placeholder = f"quantum_circuit_{circuit_index}_{line_index}"
        if len(placeholder) < len(line):
            padding = "x" * (len(line) - len(placeholder))
            placeholder += padding
        placeholder_lines.append(placeholder)
    return ", ".join(placeholder_lines)


def replace_circuit_placeholder(
    text: str, circuit_repr: str, circuit_repr_placeholder: str, circuit_drawing: str
) -> str:
    """Replace given circuit drawing placeholder w/ corresponding circuit drawing.

    Given the placeholder text for a circuit drawing, we consider each line and replace it with the
    corresponding line in the circuit drawing. In order to preserve Black's formatting, we need to
    replace the placeholder text line by line.

    Args:
        text: The output text containing a placeholder to replace.
        circuit_repr: The circuit repr (to replace the placeholder if the drawing is too large).
        circuit_repr_placeholder: The placeholder to replace with a drawing or repr.
        circuit_drawing: The single-line circuit drawing.

    Returns:
        The input text modified by replacing placeholders with drawings (or reprs).
    """
    circuit_placeholder_lines = circuit_repr_placeholder.split(", ")
    circuit_drawing_lines = get_circuit_drawing_lines(circuit_drawing, circuit_repr)

    # TODO fix bug, this isn't adding commas in the way that it should
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
        # If it's one of the last two lines (but not a newline) leave the comma. O/w remove it.
        if idx == len(circuit_placeholder_lines) - 1 and "newline" not in line:
            text = text.replace(placeholder, line)
        elif (
            idx == len(circuit_placeholder_lines) - 2
            and "newline" in circuit_placeholder_lines[idx + 1]
        ):
            text = text.replace(placeholder, line)
        else:
            text = text.replace(placeholder + ",", line)
    return text


def preprocess_compiler_output(
    compiler_output_repr: str, circuit_reprs: List[str], circuit_repr_placeholders: List[str]
) -> str:
    """Replace parts of the compiler output unrecognizable to Black with placeholders.

    Args:
        compiler_output_repr: A repr for the compiler output.
        circuit_reprs: A list of reprs for the compiler output circuits.
        circuit_repr_placeholders: A list of placeholders to replace reprs in the compiler output.

    Returns:
        A compiler output repr, preprocessed to work with Black.
    """
    # First, replace components that Black cannot parse: (a) an empty schedule will contain a
    # parenthesis followed by a comma, (b) an empty logical-to-physical qubit mapping shows up as
    # enclosed braces.
    preprocessed_out = compiler_output_repr.replace("Schedule(,", "Schedule(None,").replace(
        "{}", "empty_map"
    )
    circuit_reprs = ",|".join(circuit_reprs).split("|")
    circuit_repr_placeholders = ", newline_placeholder,|".join(circuit_repr_placeholders).split("|")
    for circuit_repr, placeholder in zip(circuit_reprs, circuit_repr_placeholders):
        preprocessed_out = preprocessed_out.replace(circuit_repr, placeholder)
    return preprocessed_out


def prettify(text: str) -> str:
    """Uses Black to format text.

    Args:
        text: A raw unformatted string.

    Returns:
        A string formatted to wrap at the hard-coded maximum line length.
    """
    return black.format_str(text, mode=black.Mode(line_length=MAX_LINE_LENGTH))


def postprocess_compiler_output(
    pretty_out: str,
    circuit_reprs: List[str],
    circuit_repr_placeholders: List[str],
    circuit_drawings: List[str],
) -> str:
    """Replace each circuit drawing placeholder with the corresponding circuit drawing line.

    Args:
        pretty_out: The prettified preprocessed compiler output repr.
        circuit_reprs: A list of reprs for the compiler output circuits.
        circuit_repr_placeholders: A list of placeholders to replace reprs in the compiler output.
        circuit_drawings: A list of circuit drawings.

    Returns:
        The final postprocessed prettified compiler output.
    """
    # First, restore preprocessed non-circuit placeholders for (a) an empty schedule will contain
    # a parenthesis followed by a comma, (b) an empty logical-to-physical qubit mapping shows up as
    # enclosed braces.
    postprocessed_out = (
        pretty_out.replace("empty_map", "{}")
        .replace("Schedule(None,", "Schedule(,")
        .replace("newline_placeholder,", "")
    )
    circuit_repr_placeholders = ", newline_placeholder,|".join(circuit_repr_placeholders).split("|")
    for circuit_repr, circuit_repr_placeholder, circuit_drawing in zip(
        circuit_reprs, circuit_repr_placeholders, circuit_drawings
    ):
        postprocessed_out = replace_circuit_placeholder(
            postprocessed_out,
            circuit_repr,
            circuit_repr_placeholder,
            circuit_drawing,
        )
    return postprocessed_out


def pretty_print_compiler_output(
    compiler_output_repr: str,
    circuit_reprs: List[str],
    circuit_drawings: List[str],
) -> str:
    """Prints human-readable version of compiler output.

    Args:
        compiler_output_repr: A repr for the compiler output.
        circuit_reprs: A list of reprs for the compiler output circuits.
        circuit_drawings: A list of circuit drawings.

    Returns
        The final postprocessed prettified compiler output.:
    """
    circuit_repr_placeholders = []
    for circuit_index, (circuit_repr, circuit_drawing) in enumerate(
        zip(circuit_reprs, circuit_drawings)
    ):
        circuit_repr_placeholders.append(
            get_circuit_placeholder(circuit_repr, circuit_drawing, circuit_index)
        )
    preprocessed_out = preprocess_compiler_output(
        compiler_output_repr, circuit_reprs, circuit_repr_placeholders
    )
    pretty_out = prettify(preprocessed_out)
    return postprocess_compiler_output(
        pretty_out, circuit_reprs, circuit_repr_placeholders, circuit_drawings
    )
