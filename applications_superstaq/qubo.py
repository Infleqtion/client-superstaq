from typing import Any, Dict, List

import numpy as np
import qubovert as qv

import applications_superstaq


def read_json_qubo_result(json_dict: dict) -> np.recarray:
    """Reads out returned JSON from SuperstaQ API's QUBO endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /qubo endpoint
    Returns:
        a numpy.recarray containing the results of the optimization.
    """
    return applications_superstaq.converters.deserialize(json_dict["solution"])


def convert_qubo_to_model(qubo: qv.QUBO) -> List[Dict[str, Any]]:
    """Takes in a qubovert QUBO and converts it to the format required by the /qubo endpoint API.
    Args:
        qubo: a qubovert QUBO object.
    Returns:
        An equivalent qubo represent as a nested list of dictionaries.
    """
    model = []
    for key, value in qubo.items():
        model.append({"keys": [str(variable) for variable in key], "value": value})
    return model


def convert_model_to_qubo(model: List[Dict[str, Any]]) -> qv.QUBO:
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
