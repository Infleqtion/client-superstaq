from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import qubovert as qv

import general_superstaq as gss
from general_superstaq import superstaq_client
from general_superstaq.typing import MaxSharpeJson, MinVolJson


@dataclass
class MinVolOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float
    qubo: qv.QUBO


def read_json_minvol(json_dict: MinVolJson) -> MinVolOutput:
    """Reads out returned JSON from SuperstaQ API's minvol endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /minvol endpoint
    Returns:
        a MinVolOutput object with the optimal portfolio.
    """
    best_portfolio = json_dict["best_portfolio"]
    best_ret = json_dict["best_ret"]
    best_std_dev = json_dict["best_std_dev"]
    qubo = gss.qubo.convert_model_to_qubo(json_dict["qubo"])
    return MinVolOutput(best_portfolio, best_ret, best_std_dev, qubo)


@dataclass
class MaxSharpeOutput:
    best_portfolio: List[str]
    best_ret: float
    best_std_dev: float
    best_sharpe_ratio: float
    qubo: qv.QUBO


def read_json_maxsharpe(json_dict: MaxSharpeJson) -> MaxSharpeOutput:
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
    qubo = gss.qubo.convert_model_to_qubo(json_dict["qubo"])
    return MaxSharpeOutput(best_portfolio, best_ret, best_std_dev, best_sharpe_ratio, qubo)


class Finance:
    def __init__(self, client: superstaq_client._SuperstaQClient):
        self._client = client

    def submit_qubo(
        self, qubo: qv.QUBO, target: str, repetitions: int = 1000
    ) -> npt.NDArray[np.int_]:
        """Submits the given QUBO to the target backend. The result of the optimization
        is returned to the user as a numpy.recarray.
        Args:
            qubo: Qubovert QUBO object representing the optimization problem.
            target: A string indicating which target to use.
            repetitions: Number of shots to execute on the device.
        Returns:
            Numpy.recarray containing the solution to the QUBO, the energy of the
            different solutions, and the number of times each solution was found.
        """
        json_dict = self._client.submit_qubo(qubo, target, repetitions=repetitions)
        return gss.qubo.read_json_qubo_result(json_dict)

    def find_min_vol_portfolio(
        self,
        stock_symbols: List[str],
        desired_return: float,
        years_window: float = 5.0,
        solver: str = "anneal",
    ) -> MinVolOutput:
        """Finds the portfolio with minimum volatility that exceeds a specified desired return.
        Args:
            stock_symbols: A list of stock tickers to pick from.
            desired_return: The minimum return needed.
            years_window: The number of years previous from today to pull data from
            for price data.
            solver: Specifies which solver to use. Defaults to a simulated annealer.
        Returns:
            MinVolOutput object, with the following attributes:
            .best_portfolio: The assets in the optimal portfolio.
            .best_ret: The return of the optimal portfolio.
            .best_std_dev: The volatility of the optimal portfolio.
        """
        input_dict: Dict[str, Union[List[str], int, float, str]] = {
            "stock_symbols": stock_symbols,
            "desired_return": desired_return,
            "years_window": years_window,
            "solver": solver,
        }
        json_dict = self._client.find_min_vol_portfolio(input_dict)
        return read_json_minvol(json_dict)

    def find_max_pseudo_sharpe_ratio(
        self,
        stock_symbols: List[str],
        k: float,
        num_assets_in_portfolio: Optional[int] = None,
        years_window: float = 5.0,
        solver: str = "anneal",
    ) -> MaxSharpeOutput:
        """
        Finds the optimal equal-weight portfolio from a possible pool of stocks
        according to the following rules:
        -All stock must come from the stock_symbols list.
        -All stocks will be equally weighted in the portfolio.
        -The "pseudo" Sharpe ratio of the portfolio is maximized.
        The Sharpe ratio can be thought of as the ratio of reward to risk.
        The formula for the Sharpe ratio is the portfolio's expected return less the risk-free
        rate divided by the portfolio standard deviation. For the risk-free rate, we will use the
        three month treasury bill rate. Instead of maximizing the Sharpe ratio directly, we will
        minimize variance minus return net the risk-free rate. The user specifies a factor k, as
        describes below to favor reducing risk or favor increasing expected return, each likely
        at the expense of the other. The Sharpe ratio of the resulting portfolio is returned,
        since it is relevant information.
        To summarize, we optimize:
        k * standard_deviation_expression - (1 - k) * expected_return_expression
        Args:
            stock_symbols: A list of stock tickers to pick from.
            k: A risk factor coefficient between 0 and 1. A k closer to 1
            indicates only being concerned with risk aversion, while a k closer to 0
            indicates only being concerned with maximizing expected return regardless of
            risk.
            k: The factor to weigh the portions of the expression.
            num_assets_in_portfolio: The number of desired assets in the portfolio.
            If not specified, then the function will iterate through and
            check for all portfolio sizes.
            years_window: The number of years previous from today to pull data from
            for price data.
            solver: Specifies which solver to use. Defaults to a simulated annealer.
        Return:
            A MaxSharpeOutput object with the following attributes:
            .best_portfolio: The assets in the optimal portfolio.
            .best_ret: The return of the optimal portfolio.
            .best_std_dev: The volatility of the optimal portfolio.
            .best_sharpe_ratio: The Sharpe ratio of the optimal portfolio.
        """
        input_dict: Dict[str, Union[List[str], float, str, int, None]] = {
            "stock_symbols": stock_symbols,
            "k": k,
            "num_assets_in_portfolio": num_assets_in_portfolio,
            "years_window": years_window,
            "solver": solver,
        }
        json_dict = self._client.find_max_pseudo_sharpe_ratio(input_dict)
        return read_json_maxsharpe(json_dict)
