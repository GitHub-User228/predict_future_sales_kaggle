from typing import Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sales_project.utils import get_bins

sns.set_theme(
    context="talk", style="darkgrid", palette="dark", font="sans-serif"
)


def linear_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    hue: str | None = None,
    figsize: Tuple[int, int] = (14, 5),
    use_index: bool = False,
    scatter: bool = False,
    linear: bool = True,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> None:
    """
    Plots a linear plot with optional scatter points for the given data.

    Args:
        data (pd.DataFrame):
            The input data frame.
        x (str):
            The column name for the x-axis.
        y (str):
            The column name for the y-axis.
        title (str):
            The title of the plot.
        hue (str, optional):
            The column name to use for coloring the lines/points.
        figsize (Tuple[int, int], optional):
            The figure size for the plot.
        use_index (bool, optional):
            Whether to use the index of the data frame for the x-axis.
        scatter (bool, optional):
            Whether to include scatter points on the plot.
    """
    plt.figure(figsize=figsize)
    if use_index:
        if linear:
            sns.lineplot(
                x=data.index, y=data[y], hue=data[hue] if hue else None
            )
        if scatter:
            sns.scatterplot(
                x=data.index, y=data[y], hue=data[hue] if hue else None
            )
    else:
        if linear:
            sns.lineplot(x=data[x], y=data[y], hue=data[hue] if hue else None)
        if scatter:
            sns.scatterplot(
                x=data[x], y=data[y], hue=data[hue] if hue else None
            )
    plt.title(title)
    plt.xlabel(x if x_label is None else x_label)
    plt.ylabel(y if y_label is None else y_label)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.show()


def decomposition_plot(
    data: pd.DataFrame,
    x: str,
) -> None:
    """
    Plots the decomposition of a time series into trend,
    seasonal, and residual components. Also checks for stationarity
    via the Augmented Dickey-Fuller test.

    Args:
        data (pd.DataFrame):
            The input data frame.
        x (str):
            The column name containing the time series data.
    """

    decomposition = seasonal_decompose(data[x], model="additive")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(14, 5 * 4), dpi=100
    )

    decomposition.observed.plot(ax=ax1, legend=False)
    ax1.set_ylabel("Observed")

    decomposition.trend.plot(ax=ax2, legend=False)
    ax2.set_ylabel("Trend")

    decomposition.seasonal.plot(ax=ax3, legend=False)
    ax3.set_ylabel("Seasonal")

    decomposition.resid.plot(ax=ax4, legend=False)
    ax4.set_ylabel("Residual")

    plt.tight_layout()
    plt.show()

    result = adfuller(decomposition.resid.dropna())
    print("-" * 30)
    print("Checking residuals for stationarity:")
    print("- ADF Statistic:", result[0])
    print("- p-value:", result[1])
    if result[1] < 0.05:
        print("Rejected the null hypothesis: the time series is stationary")
    else:
        print(
            "Can't reject the null hypothesis: the time series is non-stationary"
        )


def hist_box_plot(
    df: pd.DataFrame, feature: str, kde: bool = False, label: str | None = None
) -> None:
    """
    Plots a histogram and box plot for a given feature in a DataFrame.

    Args:
        df (pd.DataFrame):
            The input DataFrame.
        feature (str):
            The column name of the feature to plot.
        kde (bool, optional):
            Whether to plot a kernel density estimate on the histogram.
            Defaults to False.
        label (str, optional):
            The label to use for the feature.
            If not provided, the feature name will be used.
    """

    print(f"Number of observations: {len(df)}")
    print(f"Number of None: {df[feature].isnull().sum()}")
    print(f"None ratio: {df[feature].isnull().sum()/len(df)}")
    if label == None:
        label = feature
    data = df[feature].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(data, bins=get_bins(len(data)), ax=ax[0], kde=kde)
    ax[0].set_xlabel(label)
    sns.boxplot(data, ax=ax[1])
    ax[1].set_ylabel(label)
