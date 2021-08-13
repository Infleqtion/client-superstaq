import finance


def test_read_json_minvol() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    json_dict = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
    }
    assert finance.read_json_minvol(json_dict) == finance.MinVolOutput(
        best_portfolio, best_ret, best_std_dev
    )


def test_read_json_maxsharpe() -> None:
    best_portfolio = ["AAPL", "GOOG"]
    best_ret = 8.1
    best_std_dev = 10.5
    best_sharpe_ratio = 0.771
    json_dict = {
        "best_portfolio": best_portfolio,
        "best_ret": best_ret,
        "best_std_dev": best_std_dev,
        "best_sharpe_ratio": best_sharpe_ratio,
    }
    assert finance.read_json_maxsharpe(json_dict) == finance.MaxSharpeOutput(
        best_portfolio, best_ret, best_std_dev, best_sharpe_ratio
    )
