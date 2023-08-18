from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from sklearn.linear_model import LinearRegression


def plot_results(
    scores: List[float], tick_labels: List[str], savefn: Optional[str] = None, show: bool = True
) -> None:
    """Plot a simple bar chart of the benchmark results.

    The ordering of scores and labels is assumed to be the same.

    Args:
        scores: List of benchmark results.
        tick_labels: List of quantum device names.
        savefn: Path to save the plot, if `None`, the plot is not saved.
        show: Display the plot using `plt.show`.
    """
    _, ax = plt.subplots(dpi=150)

    width = 0.4

    xvals = np.arange(len(scores))

    ax.bar(xvals, scores, width=width, tick_label=tick_labels, align="center")

    ax.set_xticks(range(len(scores)))
    ax.set_ylabel("Score")
    plt.tight_layout()

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover

    plt.close()


def plot_volumetric_results(
    benchmark_data: Iterable[Tuple[int, int, float]],
    savefn: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    xmax: float = 60,
    ymax: float = 7,
    rect_width: float = 1.5,
    rect_height: float = 0.2,
) -> None:
    """Plot the benchmark results on an (x, y) = (depth, qubits) axis.

    It is assumed that all of the given data was collected on a single device. Keyword arguments
    may need to be tweaked depending on the domain and range of benchmark data.

    Args:
        benchmark_data: List of tuples containing (circuit depth, qubits, score) for each benchmark
            executed on a single device.
        savefn: Path to save the plot, if `None`, the plot is not saved.
        title: Optional title for the plot.
        show: Display the plot using `plt.show`.
        xmax: The rightmost limit of the x axis.
        ymax: The uppermost limit of the y axis.
        rect_width: The width of the rectangles.
        rect_height: The height of the rectangles.
    """
    _, ax = plt.subplots(figsize=[4, 4])

    cmap = matplotlib.cm.RdBu
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Benchmark Score")

    for x, y, z in benchmark_data:
        rect = plt.Rectangle(
            (x - rect_width / 2, y - rect_height / 2),
            rect_width,
            rect_height,
            color=cmap(norm(z)),
        )
        ax.add_patch(rect)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Qubits")
    if title:
        ax.set_title(title)

    if savefn:
        # Don't want to save figures when running tests
        plt.savefig(savefn, bbox_inches="tight")  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover
    plt.close()


def plot_correlations(
    benchmark_features: Dict[str, List[float]],
    device_scores: Union[Dict[str, float], Iterable[Dict[str, float]]],
    feature_labels: List[str],
    device_name: Union[str, List[str]],
    savefn: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot a correlation heatmap of the features for one or multiple devices.

    Args:
        benchmark_features: A dictionary where the keys are benchmark names and the values are the
            list of feature values for that benchmark.
        device_scores: A dictionary of (benchmark name, score) pairs, or a list of such
            dictionaries.
        feature_labels: Feature names, should have the same length as the lists of feature values
            in `benchmark_features`.
        device_name: The name or list of names of quantum device(s) where the scores were obtained.
        savefn: Path to save the plot, if `None`, the plot is not saved.
        show: Display the plot using `plt.show`.
    """

    temp_correlations = []
    if isinstance(device_scores, Dict):
        device_scores = [device_scores]
    for cur_device_scores in device_scores:
        device_correlations = []
        for i in range(len(feature_labels)):
            x, y = [], []
            for benchmark in cur_device_scores:
                x.append(benchmark_features[benchmark][i])
                y.append(cur_device_scores[benchmark])

            proper_x = np.array(x)[:, np.newaxis]
            proper_y = np.array(y)
            model = LinearRegression().fit(proper_x, proper_y)
            correlation = model.score(proper_x, proper_y)
            device_correlations.append(correlation)
        temp_correlations.append(device_correlations)

    correlations = np.array(temp_correlations)

    if isinstance(device_name, str):
        device_names = [device_name]
    else:
        device_names = device_name

    _, ax = plt.subplots(dpi=300)
    im, _ = heatmap(
        correlations,
        ax,
        device_names,
        feature_labels,
        cmap="cool",
        vmin=0,
        cbarlabel=r"Coefficient of Determination, $R^2$",
        cbar_kw={"pad": 0.005},
    )

    annotate_heatmap(im, size=7)

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn, bbox_inches="tight")  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover
    plt.close()


def plot_benchmark(
    data: List[Union[str, List[str], List[List[float]]]],
    show: bool = True,
    savefn: Optional[str] = None,
    spoke_labels: Optional[List[str]] = None,
    legend_loc: Tuple[float, float] = (0.75, 0.85),
) -> None:
    """Create a radar plot showing the feature vectors of the given benchmarks.

    Args:
        data: Contains the title, feature data, and labels in the format:
            [title, [benchmark labels], [[features_1], [features_2], ...]].
        show: Display the plot using `plt.show`.
        savefn: Path to save the plot, if `None`, the plot is not saved.
        spoke_labels: Optional labels for the feature vector dimensions.
        legend_loc: Optional argument to fine tune the legend placement.
    """
    if spoke_labels is None:
        spoke_labels = ["Connectivity", "Liveness", "Parallelism", "Measurement", "Entanglement"]

    num_spokes = len(spoke_labels)
    theta = radar_factory(num_spokes)

    _, ax = plt.subplots(dpi=150, subplot_kw=dict(projection="radar"))

    title, labels, case_data = data
    ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(
        title,
        weight="bold",
        size="medium",
        position=(0.5, 1.1),
        horizontalalignment="center",
        verticalalignment="center",
    )
    for d, label in zip(case_data, labels):
        ax.plot(theta, d, label=label)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    ax.legend(loc=legend_loc, labelspacing=0.1, fontsize=11)
    plt.tight_layout()

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover

    plt.close()


def heatmap(
    data: npt.NDArray[np.float_],
    ax: matplotlib.axes.Axes,
    row_labels: List[str],
    col_labels: List[str],
    cbar_kw: Optional[Dict[str, Any]] = None,
    cbarlabel: str = "",
    **kwargs: Any
) -> Tuple[matplotlib.image.AxesImage, Any]:
    """Create a heatmap from a numpy array and two lists of labels.

    Args:
        data: A 2D numpy array of shape (N, M).
        row_labels: A list or array of length N with the labels for the rows.
        col_labels: A list or array of length M with the labels for the columns.
        ax: An optional `matplotlib.axes.Axes` instance to which the heatmap is plotted.
            If not provided, use current axes or create a new one.
        cbar_kw: An optional dictionary with arguments to `matplotlib.Figure.colorbar`.
        cbarlabel: An optional label for the colorbar.
        kwargs: All other arguments are forwarded to `imshow`.

    Returns:
        The generated heatmap and the associated color bar.
    """
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar_kw is None:
        cbar_kw = {}  # pragma: no cover
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", **cbar_kw)
    cbar.ax.set_xlabel(cbarlabel, labelpad=10, va="bottom", fontsize=8)
    cbar.ax.tick_params(labelsize=10)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=8)  # regular fontsize is 12
    ax.set_yticklabels(row_labels, fontsize=7)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the 3 typical features separately
    for t in ax.get_xticklabels()[-3:]:
        t.set_horizontalalignment("left")
        t.set_rotation(35)

    return im, cbar


def annotate_heatmap(
    im: matplotlib.image.AxesImage,
    data: Optional[npt.NDArray[np.float_]] = None,
    valfmt: Any = "{x:.2f}",
    textcolors: Tuple[str, str] = ("black", "white"),
    threshold: Optional[float] = None,
    **textkw: Any
) -> List[matplotlib.text.Text]:
    """Annotate the given heatmap.

    Args:
        im: The `AxesImage` to be labeled.
        data: Optional data used to annotate. If None, the image's data is used.
        valfmt: An optional format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.
        textcolors: A pair of colors.  The first is used for values below a threshold, the second
            for those above. Defaults to black and white respectively.
        threshold: An optional value in data units according to which of the colors from textcolors
            are applied. If None (the default) uses the middle of the colormap as separation.
        textkw: All other arguments are forwarded to each call to `text` used to create
            the text labels.

    Returns:
        List of the text annotations.
    """

    if data is None:
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)  # pragma: no cover
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def radar_factory(num_vars: int) -> npt.NDArray[np.float_]:
    """Create a radar chart with `num_vars` axes.

    (https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html) This function
    creates a `RadarAxes` projection and registers it.

    Args:
        num_vars: Number of variables for radar chart.

    Returns:
        A list of evenly spaced angles.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(RadarAxesMeta):
        """A helper class that sets the shape of the feature plot"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initializes the helper `RadarAxes` class."""
            self.frame = "circle"
            self.theta = theta
            self.num_vars = num_vars
            super().__init__(*args, **kwargs)

    register_projection(RadarAxes)
    return theta


class RadarAxesMeta(PolarAxes):
    """A helper class to generate feature-vector plots."""

    name = "radar"
    # use 1 line segment to connect specified points
    RESOLUTION = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the `RadarAxesMeta` class."""
        super().__init__(*args, **kwargs)
        # rotate plot such that the first axis is at the top
        self.set_theta_zero_location("N")

    def fill(
        self, *args: Any, closed: bool = True, **kwargs: Any
    ) -> List[matplotlib.patches.Polygon]:
        """Method to override fill so that line is closed by default.

        Args:
            args: Arguments to be passed to fill.
            closed: Optional parameter to close fill or not. Defaults to True.
            kwargs: Other desired keyworded arguments to be passed to fill.

        Returns:
            A list of `matplotlib.patches.Polygon`.
        """
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Overrides plot so that line is closed by default.

        Args:
            args: Desired arguments for plotting.
            kwargs: Other desired keyword arguments for plotting.
        """
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line: matplotlib.lines.Line2D) -> None:
        """A method to close the input line.

        Args:
            line: The line to close.
        """
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up.
        # See issue https://github.com/Infleqtion/client-superstaq/issues/27
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels: List[str]) -> None:
        """Sets the spoke labels at the appropriate points on the radar plot.

        Args:
            labels: The list of labels to apply.
        """
        self.set_thetagrids(np.degrees(self.theta), labels, fontsize=14)

    def _gen_axes_patch(self) -> matplotlib.patches.Circle:
        # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
        # in axes coordinates.
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self) -> matplotlib.spines.Spine:
        return super()._gen_axes_spines()
