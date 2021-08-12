from dataclasses import dataclass
from typing import List


@dataclass
class MinVolOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float


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
    return MinVolOutput(best_portfolio, best_ret, best_std_dev)


@dataclass
class MaxSharpeOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float
    best_sharpe_ratio: float


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
    return MaxSharpeOutput(best_portfolio, best_ret, best_std_dev, best_sharpe_ratio)
