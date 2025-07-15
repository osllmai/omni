#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Partial dependence plots.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure
from collections import OrderedDict


class PDPExplanation(ExplanationBase):
    """
    The class for PDP explanation results. The PDP explanation results are stored in a dict.
    The key in the dict is "global" indicating PDP is a global explanation method.
    The value in the dict is another dict with the following format:
    `{feature_name: {"values": the PDP grid values, "scores": the average PDP scores,
    "sampled_scores": the PDP scores computed with Monte-Carlo samples}}`.
    """

    def __init__(self, mode):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.mode = mode
        self.explanations = OrderedDict()

    def __repr__(self):
        return repr(self.explanations)

    def add(self, feature_name, values, scores, sampled_scores=None):
        """
        Adds the raw values of the partial dependence function
        corresponding to one specific feature.

        :param feature_name: The feature column name.
        :param values: The features values.
        :param scores: The average PDP scores corresponding to the values.
        :param sampled_scores: The PDP scores computed with Monte-Carlo samples.
        """
        self.explanations[feature_name] = {
            "values": values,
            "scores": scores,
            "sampled_scores": sampled_scores,
        }

    def get_explanations(self):
        """
        Gets the partial dependence scores.

        :return: A dict containing the partial
            dependence scores of all the studied features with the following format:
            `{feature_name: {"values": the feature values, "scores": the average PDP scores,
            "sampled_scores": the PDP scores computed with Monte-Carlo samples}}`.
        """
        return self.explanations

    def plot(self, class_names=None, **kwargs):
        """
        Returns a matplotlib figure showing the PDP explanations.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A matplotlib figure plotting PDP explanations.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations()
        features = list(explanations.keys())

        if not features:
            return []

        # Calculate grid dimensions for better layout - fewer columns for more space
        n_features = len(features)
        if n_features <= 2:
            num_cols = n_features
            num_rows = 1
        elif n_features <= 4:
            num_cols = 2
            num_rows = 2
        else:
            num_cols = 2  # Max 2 columns for better readability
            num_rows = int(np.ceil(n_features / 2))

        # Create figure with generous sizing
        fig_width = max(20, num_cols * 10)  # Much wider
        fig_height = max(12, num_rows * 6)  # Much taller
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False
        )

        for i, feature in enumerate(features):
            row, col = divmod(i, num_cols)
            exp = explanations[feature]
            plt.sca(axes[row, col])
            values = [self._s(v, max_len=10) for v in exp["values"]]  # Shorter labels
            plt.plot(values, exp["scores"], linewidth=2)

            # Better handling of categorical features
            if isinstance(values[0], str):
                # For categorical features, reduce number of ticks if too many
                if len(values) > 10:
                    step = len(values) // 8
                    tick_indices = list(range(0, len(values), step))
                    plt.xticks(
                        tick_indices,
                        [values[j] for j in tick_indices],
                        rotation=45,
                        ha="right",
                    )
                else:
                    plt.xticks(rotation=45, ha="right")
            else:
                # For numerical features, use fewer ticks
                plt.locator_params(axis="x", nbins=6)

            plt.ylabel("Partial dependence", fontsize=12)
            plt.title(feature, fontsize=14, fontweight="bold", pad=20)

            # Better legend positioning
            if class_names is not None:
                plt.legend(class_names, fontsize=10, loc="best", framealpha=0.9)
            else:
                if self.mode == "classification":
                    plt.legend(
                        [f"Class {i}" for i in range(exp["scores"].shape[1])],
                        fontsize=10,
                        loc="best",
                        framealpha=0.9,
                    )
                else:
                    plt.legend(["Target"], fontsize=10, loc="best", framealpha=0.9)
            plt.grid(alpha=0.3)

            # Improve tick label sizes
            plt.tick_params(axis="both", which="major", labelsize=10)

            if exp["sampled_scores"] is not None:
                for scores in exp["sampled_scores"]:
                    plt.plot(values, scores, color="#808080", alpha=0.1, linewidth=1)

        # Hide unused subplots
        for i in range(n_features, num_rows * num_cols):
            row, col = divmod(i, num_cols)
            axes[row, col].set_visible(False)

        # Better layout with more spacing
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        return [fig]

    def _plotly_figure(self, class_names=None, **kwargs):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        explanations = self.get_explanations()
        features = list(explanations.keys())

        # Use same layout logic as matplotlib version - max 2 columns for readability
        n_features = len(features)
        if n_features <= 2:
            num_cols = n_features
            num_rows = 1
        elif n_features <= 4:
            num_cols = 2
            num_rows = 2
        else:
            num_cols = 2  # Max 2 columns for better readability
            num_rows = int(np.ceil(n_features / 2))
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=features)
        for i, feature in enumerate(features):
            e = explanations[feature]
            row, col = divmod(i, num_cols)
            values = [
                self._s(v, max_len=8) for v in e["values"]
            ]  # Shorter labels for better display
            if self.mode == "classification":
                for k in range(e["scores"].shape[1]):
                    label = class_names[k] if class_names is not None else f"Class {k}"
                    fig.add_trace(
                        go.Scatter(
                            x=values,
                            y=e["scores"][:, k],
                            name=self._s(str(feature), 10),
                            legendgroup=label,
                            legendgrouptitle_text=label,
                        ),
                        row=row + 1,
                        col=col + 1,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=e["scores"].flatten(),
                        name=self._s(str(feature), 10),
                        legendgroup="Target",
                    ),
                    row=row + 1,
                    col=col + 1,
                )

            if e["sampled_scores"] is not None:
                for scores in e["sampled_scores"]:
                    if self.mode == "classification":
                        for k in range(scores.shape[1]):
                            label = (
                                class_names[k]
                                if class_names is not None
                                else f"Class {k}"
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=values,
                                    y=scores[:, k],
                                    opacity=0.1,
                                    mode="lines",
                                    showlegend=False,
                                    line=dict(color="#808080"),
                                    legendgroup=label,
                                ),
                                row=row + 1,
                                col=col + 1,
                            )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=values,
                                y=scores.flatten(),
                                opacity=0.1,
                                mode="lines",
                                showlegend=False,
                                line=dict(color="#808080"),
                                legendgroup="Target",
                            ),
                            row=row + 1,
                            col=col + 1,
                        )
        # Set generous figure dimensions for better readability
        fig_height = max(600, num_rows * 400)  # Much taller charts
        fig_width = max(1200, num_cols * 600)  # Much wider charts
        fig.update_layout(
            height=fig_height, width=fig_width, title_x=0.5, font=dict(size=12)
        )
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the PDP explanations.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting PDP explanations.
        """
        return DashFigure(self._plotly_figure(class_names=class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Shows the partial dependence plots in IPython.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names=class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        explanations = {}
        for name, e in d["explanations"].items():
            e["values"] = np.array(e["values"])
            e["scores"] = np.array(e["scores"])
            e["sampled_scores"] = (
                np.array(e["sampled_scores"])
                if e["sampled_scores"] is not None
                else None
            )
            explanations[name] = e
        exp = PDPExplanation(mode=d["mode"])
        exp.explanations = explanations
        return exp
