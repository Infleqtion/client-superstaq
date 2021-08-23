from dataclasses import dataclass
from typing import List

import qubovert as qv

import applications_superstaq


@dataclass
class MinVolOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float
    qubo: qv.QUBO


def read_json_minvol(json_dict: dict) -> MinVolOutput:
    """Reads out returned JSON from SuperstaQ API's minvol endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /minvol endpoint
    Returns:
        a MinVolOutput object with the optimal portfolio.
    """

    best_portfolio = json_dict["best_portfolio"]
    best_ret = json_dict["best_ret"]
    best_std_dev = json_dict["best_std_dev"]
    qubo = applications_superstaq.qubo.convert_model_to_qubo(json_dict["qubo"])
    return MinVolOutput(best_portfolio, best_ret, best_std_dev, qubo)


@dataclass
class MaxSharpeOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float
    best_sharpe_ratio: float
    qubo: qv.QUBO


def read_json_maxsharpe(json_dict: dict) -> MaxSharpeOutput:
    """Reads out returned JSON from SuperstaQ API's minvol endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /maxsharpe endpoint
    Returns:
        a MaxSharpeOutput object with the optimal portfolio.
    """

    best_portfolio = json_dict["best_portfolio"]
    best_ret = json_dict["best_ret"]
    best_std_dev = json_dict["best_std_dev"]
    best_sharpe_ratio = json_dict["best_sharpe_ratio"]
    qubo = applications_superstaq.qubo.convert_model_to_qubo(json_dict["qubo"])
    return MaxSharpeOutput(best_portfolio, best_ret, best_std_dev, best_sharpe_ratio, qubo)
