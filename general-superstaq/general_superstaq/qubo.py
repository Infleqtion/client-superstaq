from typing import Dict, List, Tuple

import general_superstaq as gss


def read_json_qubo_result(json_dict: Dict[str, str]) -> List[Dict[Tuple[int], int]]:
    """Reads out returned JSON from Superstaq API's QUBO endpoint.

    Args:
        json_dict: A JSON dictionary matching the format returned by the /qubo endpoint.

    Returns:
        A `numpy.recarray` containing the results of the optimization.
    """
    return gss.serialization.deserialize(json_dict["solution"])
