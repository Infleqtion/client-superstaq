import supermarq as sm


def test_plot_benchmark() -> None:
    sm.feature_plotting.plot_benchmark(
        ["test title", ["b1", "b2", "b3"], [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]],
        spoke_labels=["f1", "f2", "f3"],
        show=False,
    )
