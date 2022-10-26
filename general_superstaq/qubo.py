from typing import Dict

import numpy as np
import numpy.typing as npt
import qubovert as qv

import general_superstaq as gss
from general_superstaq.typing import QuboModel


def read_json_qubo_result(json_dict: Dict[str, str]) -> npt.NDArray[np.int_]:
    """Reads out returned JSON from SuperstaQ API's QUBO endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /qubo endpoint
    Returns:
        a numpy.recarray containing the results of the optimization.
    """
    return gss.serialization.deserialize(json_dict["solution"])


def convert_qubo_to_model(qubo: qv.QUBO) -> QuboModel:
    """Takes in a qubovert QUBO and converts it to the format required by the /qubo endpoint API.
    Args:
        qubo: a qubovert QUBO object.
    Returns:
        An equivalent qubo represent as a nested list of dictionaries.
    """
    model: QuboModel = []
    for key, value in qubo.items():
        model.append({"keys": [str(variable) for variable in key], "value": value})
    return model


def convert_model_to_qubo(model: QuboModel) -> qv.QUBO:
    """Takes in qubo model transferred over the wire and converts it to the qubovert format.
    Args:
        model: The qubo model as specified in superstaq.web.server.
    Returns:
        An equivalent qubovert.QUBO object.
    """
    qubo_dict = {}
    for term in model:
        qubo_dict[tuple(term["keys"])] = term["value"]
    return qv.QUBO(qubo_dict)
