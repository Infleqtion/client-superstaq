from typing import Dict

import numpy as np
import numpy.typing as npt
import qubovert as qv

import general_superstaq as gss


def read_json_qubo_result(json_dict: Dict[str, str]) -> npt.NDArray[np.int_]:
    """Reads out returned JSON from Superstaq API's QUBO endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by the /qubo endpoint.

    Returns:
        A `numpy.recarray` containing the results of the optimization.
    """
    return gss.serialization.deserialize(json_dict["solution"])


def convert_qubo_to_model(qubo: qv.QUBO) -> gss.QuboModel:
    """Takes in a qubovert QUBO and converts it to the format required by the /qubo endpoint API.

    Args:
        qubo: A `qubovert.QUBO` object.

    Returns:
        An equivalent qubo represented as a nested list of dictionaries.
    """
    model: gss.QuboModel = []
    for key, value in qubo.items():
        model.append({"keys": [str(variable) for variable in key], "value": value})
    return model
