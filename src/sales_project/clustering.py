import cudf
import pandas as pd
import seaborn as sns
from typing import Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sales_project.plotters import linear_plot
from sklearn.cluster import AgglomerativeClustering
from cuml.cluster import AgglomerativeClustering as cuAggCl

sns.set_theme(
    context="talk", style="darkgrid", palette="dark", font="sans-serif"
)


def agglomerative_clustering(
    df: pd.DataFrame | cudf.DataFrame,
    features: list[str],
    n_clusters_min: int,
    n_clusters_max: int,
    n_clusters_step: int,
    metric: str,
    cluster_feat_name: str,
    clusters_fig_size: Tuple[10, 10] = (10, 10),
) -> pd.DataFrame:
    """
    Performs agglomerative clustering on the input DataFrame and
    returns the DataFrame with the cluster labels.

    Args:
        df (pd.DataFrame | cudf.DataFrame):
            The input DataFrame to perform clustering on.
        features (list[str]):
            The list of feature columns to use for clustering.
        n_clusters_min (int):
            The minimum number of clusters to try.
        n_clusters_max (int):
            The maximum number of clusters to try.
        n_clusters_step (int):
            The step size for the number of clusters to try.
        metric (str):
            The distance metric to use for clustering.
        cluster_feat_name (str):
            The name of the column to store the cluster labels.
        clusters_fig_size (Tuple[10, 10]):
            The size of the figure to plot the clusters.

    Returns:
        pd.DataFrame:
            The input DataFrame with the cluster labels added.
    """
    df_scores = pd.DataFrame(columns=["n_clusters", "silhouette_score"])
    if type(df) == cudf.DataFrame:
        for it, n_clusters in tqdm(
            enumerate(
                range(n_clusters_min, n_clusters_max + 1, n_clusters_step)
            )
        ):
            model = cuAggCl(n_clusters=n_clusters, metric=metric)
            labels = model.fit_predict(df[features]).to_numpy()
            silhouette_score_ = silhouette_score(
                df[features].to_pandas(), labels
            )
            df_scores.loc[it] = (n_clusters, silhouette_score_)
        n_clusters = df_scores[
            df_scores["silhouette_score"]
            == df_scores["silhouette_score"].max()
        ].n_clusters.item()
        model = cuAggCl(n_clusters=n_clusters, metric=metric)
    elif type(df) == pd.DataFrame:
        for it, n_clusters in tqdm(
            enumerate(
                range(n_clusters_min, n_clusters_max + 1, n_clusters_step)
            )
        ):
            model = AgglomerativeClustering(
                n_clusters=n_clusters, metric=metric
            )
            labels = model.fit_predict(df[features])
            silhouette_score_ = silhouette_score(df[features], labels)
            df_scores.loc[it] = (n_clusters, silhouette_score_)
        n_clusters = df_scores[
            df_scores["silhouette_score"]
            == df_scores["silhouette_score"].max()
        ].n_clusters.item()
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric)
    df[cluster_feat_name] = model.fit_predict(df[features])
    linear_plot(df_scores, "n_clusters", "silhouette_score", "")
    plt.figure(figsize=clusters_fig_size)
    sns.scatterplot(
        data=df[features + [cluster_feat_name]].to_pandas(),
        x=features[0],
        y=features[1],
        hue=cluster_feat_name,
    )
    return df
