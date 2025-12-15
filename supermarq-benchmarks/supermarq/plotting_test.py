# Copyright 2025 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import supermarq


def test_plot_benchmark() -> None:
    supermarq.plotting.plot_benchmark(
        "test title",
        ["b1", "b2", "b3"],
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        spoke_labels=["f1", "f2", "f3"],
        show=False,
    )

    supermarq.plotting.plot_benchmark(
        "test title",
        ["b1", "b2", "b3", "b4", "b5"],
        [[0.1] * 5, [0.2] * 5, [0.3] * 5, [0.4] * 5, [0.5] * 5],
        spoke_labels=None,
        show=False,
    )


def test_plot_volumetric_results() -> None:
    supermarq.plotting.plot_volumetric_results(
        [(12, 6, 0.5), (20, 20, 0.01)],
        ymax=100,
        xmax=50,
        rect_width=0.2,
        rect_height=0.2,
        title="test",
        show=False,
    )


def test_plot_results() -> None:
    supermarq.plotting.plot_results([0.1, 0.2], ["b1", "b2"], show=False)


def test_plot_correlations() -> None:
    features = {"ghz5": [0.4, 1.0, 0.8, 0.47, 0.0, 0], "hsim4": [0.5, 1.0, 0.285, 0.59, 0.0, 0.38]}
    scores = {"ghz5": 1.0, "hsim4": 0.1}

    # Test with a single device
    supermarq.plotting.plot_correlations(
        features,
        scores,
        ["PC", "CD", "Ent", "Liv", "Mea", "Par"],
        device_name="ibmq_sim",
        show=False,
    )

    # Test with multiple devices
    supermarq.plotting.plot_correlations(
        features,
        [scores, scores],
        ["PC", "CD", "Ent", "Liv", "Mea", "Par"],
        device_name=["ibmq_sim", "aws_sim"],
        show=False,
    )
