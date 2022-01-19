from unittest import mock

import numpy as np
import qubovert as qv

import applications_superstaq


def test_read_json_minvol() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
        "qubo": applications_superstaq.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert applications_superstaq.finance.read_json_minvol(
        json_dict
    ) == applications_superstaq.finance.MinVolOutput(
        best_portfolio, best_ret, best_std_dev, qubo_obj
    )


def test_read_json_maxsharpe() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    best_sharpe_ratio = 0.771
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
        "best_sharpe_ratio": best_sharpe_ratio,
        "qubo": applications_superstaq.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert applications_superstaq.finance.read_json_maxsharpe(
        json_dict
    ) == applications_superstaq.finance.MaxSharpeOutput(
        best_portfolio, best_ret, best_std_dev, best_sharpe_ratio, qubo_obj
    )


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.submit_qubo",
    return_value={
        "solution": applications_superstaq.converters.serialize(
            np.rec.fromrecords(
                [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
                dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
            )
        )
    },
)
def test_service_submit_qubo(mock_submit_qubo: mock.MagicMock) -> None:
    client = applications_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="applications_superstaq"
    )
    service = applications_superstaq.finance.Finance(client)
    expected = np.rec.fromrecords(
        [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
        dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
    )
    assert repr(service.submit_qubo(qv.QUBO(), "target", repetitions=10)) == repr(expected)


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.find_min_vol_portfolio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_min_vol_portfolio(mock_find_min_vol_portfolio: mock.MagicMock) -> None:
    client = applications_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="applications_superstaq"
    )
    service = applications_superstaq.finance.Finance(client)
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MinVolOutput(["AAPL", "GOOG"], 8.1, 10.5, qubo)
    assert service.find_min_vol_portfolio(["AAPL", "GOOG", "IEF", "MMM"], 8) == expected


@mock.patch(
    "applications_superstaq.superstaq_client._SuperstaQClient.find_max_pseudo_sharpe_ratio",
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
    client = applications_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="applications_superstaq"
    )
    service = applications_superstaq.finance.Finance(client)
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MaxSharpeOutput(
        ["AAPL", "GOOG"], 8.1, 10.5, 0.771, qubo
    )
    assert service.find_max_pseudo_sharpe_ratio(["AAPL", "GOOG", "IEF", "MMM"], k=0.5) == expected
