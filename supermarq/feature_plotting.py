from typing import Any, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes


def plot_benchmark(
    data: List[Union[str, List[str], List[List[float]]]],
    show: bool = True,
    savefn: Optional[str] = None,
    spoke_labels: Optional[List[str]] = None,
    legend_loc: Tuple[float, float] = (0.75, 0.85),
) -> None:
    """
    Create a radar plot of the given benchmarks.

    Input
    -----
    data : Contains the title, feature data, and labels in the format:
        [title, [benchmark labels], [[features_1], [features_2], ...]]
    """
    plt.rcParams["font.family"] = "Times New Roman"

    if spoke_labels is None:
        spoke_labels = ["Connectivity", "Liveness", "Parallelism", "Measurement", "Entanglement"]

    num_spokes = len(spoke_labels)
    theta = radar_factory(num_spokes)

    _, ax = plt.subplots(dpi=150, subplot_kw=dict(projection="radar"))

    _, labels, case_data = data
    ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
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


def radar_factory(num_vars: int) -> np.ndarray:
    """
    (https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)

    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(RadarAxesMeta):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.frame = "circle"
            self.theta = theta
            self.num_vars = num_vars
            super().__init__(*args, **kwargs)

    register_projection(RadarAxes)
    return theta


class RadarAxesMeta(PolarAxes):

    name = "radar"
    # use 1 line segment to connect specified points
    RESOLUTION = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # rotate plot such that the first axis is at the top
        self.set_theta_zero_location("N")

    def fill(
        self, *args: Any, closed: bool = True, **kwargs: Any
    ) -> List[matplotlib.patches.Polygon]:
        """Override fill so that line is closed by default"""
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Override plot so that line is closed by default"""
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line: matplotlib.lines.Line2D) -> None:
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up. See issue https://github.com/SupertechLabs/SupermarQ/issues/27
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels: List[str]) -> None:
        self.set_thetagrids(np.degrees(self.theta), labels, fontsize=14)

    def _gen_axes_patch(self) -> matplotlib.patches.Circle:
        # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
        # in axes coordinates.
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self) -> matplotlib.spines.Spine:
        return super()._gen_axes_spines()
