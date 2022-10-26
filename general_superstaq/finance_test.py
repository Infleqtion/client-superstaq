from unittest import mock

import numpy as np
import qubovert as qv

import general_superstaq as gss
from general_superstaq.typing import MaxSharpeJson, MinVolJson


def test_read_json_minvol() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict: MinVolJson = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
        "qubo": gss.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert gss.finance.read_json_minvol(json_dict) == gss.finance.MinVolOutput(
        best_portfolio, best_ret, best_std_dev, qubo_obj
    )


def test_read_json_maxsharpe() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    best_sharpe_ratio = 0.771
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict: MaxSharpeJson = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
        "best_sharpe_ratio": best_sharpe_ratio,
        "qubo": gss.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert gss.finance.read_json_maxsharpe(json_dict) == gss.finance.MaxSharpeOutput(
        best_portfolio, best_ret, best_std_dev, best_sharpe_ratio, qubo_obj
    )


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.submit_qubo",
    return_value={
        "solution": gss.serialization.serialize(
            np.rec.fromrecords(
                [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
                dtype=[
                    ("solution", "O"),
                    ("energy", "<f8"),
                    ("num_occurrences", "<i8"),
                ],
            )
        )
    },
)
def test_service_submit_qubo(mock_submit_qubo: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.finance.Finance(client)
    expected = np.rec.fromrecords(
        [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
        dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
    )
    assert repr(service.submit_qubo(qv.QUBO(), "target", repetitions=10)) == repr(expected)


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.find_min_vol_portfolio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_min_vol_portfolio(mock_find_min_vol_portfolio: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.finance.Finance(client)
    qubo = {("0",): 123}
    expected = gss.finance.MinVolOutput(["AAPL", "GOOG"], 8.1, 10.5, qubo)
    assert service.find_min_vol_portfolio(["AAPL", "GOOG", "IEF", "MMM"], 8) == expected


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.find_max_pseudo_sharpe_ratio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "best_sharpe_ratio": 0.771,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_max_pseudo_sharpe_ratio(
    mock_find_max_pseudo_sharpe_ratio: mock.MagicMock,
) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.finance.Finance(client)
    qubo = {("0",): 123}
    expected = gss.finance.MaxSharpeOutput(["AAPL", "GOOG"], 8.1, 10.5, 0.771, qubo)
    assert service.find_max_pseudo_sharpe_ratio(["AAPL", "GOOG", "IEF", "MMM"], k=0.5) == expected
