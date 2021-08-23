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
