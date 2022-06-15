import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes


def plot_benchmark(data, show=True, savefn=None, spoke_labels=None, legend_loc=(0.75, 0.85)):
    """
    Create a radar plot of the given benchmarks.

    Input
    -----
    data : List[str, List[str], List[float]]
        Contains the title, feature data, and labels in the format:
        [title, [labels], [feature vecs: [con, liv, par, mea, ent] ]]
    """
    plt.rcParams["font.family"] = "Times New Roman"

    if spoke_labels is None:
        spoke_labels = ["Connectivity", "Liveness", "Parallelism", "Measurement", "Entanglement"]

    N = len(spoke_labels)
    theta = radar_factory(N, frame="circle")

    fig, ax = plt.subplots(dpi=150, subplot_kw=dict(projection="radar"))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    title, labels, case_data = data
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #             horizontalalignment='center', verticalalignment='center')
    for d, label in zip(case_data, labels):
        ax.plot(theta, d, label=label)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    ax.legend(loc=legend_loc, labelspacing=0.1, fontsize=11)
    plt.tight_layout()

    if savefn is not None:
        plt.savefig(savefn)

    if show:
        plt.show()

    plt.close()


def radar_factory(num_vars, frame="circle"):
    """
    (https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)

    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(RadarAxesMeta):
        def __init__(self, *args, **kwargs):
            self.frame = frame
            self.theta = theta
            self.num_vars = num_vars
            super().__init__(*args, **kwargs)

    register_projection(RadarAxes)
    return theta


class RadarAxesMeta(PolarAxes):

    name = "radar"
    # use 1 line segment to connect specified points
    RESOLUTION = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rotate plot such that the first axis is at the top
        self.set_theta_zero_location("N")

    def fill(self, *args, closed=True, **kwargs):
        """Override fill so that line is closed by default"""
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """Override plot so that line is closed by default"""
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line):
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(self.theta), labels, fontsize=14)

    def _gen_axes_patch(self):
        # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
        # in axes coordinates.
        if self.frame == "circle":
            return Circle((0.5, 0.5), 0.5)
        else:
            raise ValueError("Unknown value for 'frame': %s" % self.frame)

    def _gen_axes_spines(self):
        if self.frame == "circle":
            return super()._gen_axes_spines()
        else:
            raise ValueError("Unknown value for 'frame': %s" % self.frame)
