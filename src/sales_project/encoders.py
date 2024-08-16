import cudf
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from cuml.manifold import TSNE as TSNE_cuml
from sentence_transformers import SentenceTransformer


def encode(
    df: cudf.DataFrame | pd.DataFrame,
    column: str,
    n_components: int = 2,
    perplexity: int = 30,
    is_gpu_accelerated: bool = False,
    tsne_method: str = "exact",
):
    """
    Encodes the specified column in the input DataFrame using
    a pre-trained sentence transformer model and projects the
    embeddings to a lower dimensional space using t-SNE.

    Args:
        df (cudf.DataFrame | pd.DataFrame):
            The input DataFrame containing the column to be encoded.
        column (str):
            The name of the column to be encoded.
        n_components (int, optional):
            The number of dimensions to reduce the embeddings to.
            Defaults to 2.
        perplexity (int, optional):
            The perplexity parameter for the t-SNE algorithm.
            Defaults to 30.
        is_gpu_accelerated (bool, optional):
            Whether to use the GPU-accelerated t-SNE implementation from cuML.
            Defaults to False.
        tsne_method (str, optional):
            The t-SNE algorithm to use, either "exact" or "barnes_hut".
            Defaults to "exact".

    Returns:
        cudf.DataFrame | pd.DataFrame:
            The input DataFrame with the encoded column appended.
    """
    with torch.no_grad():
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        embeddings = model.encode(
            df[column].to_arrow().to_pylist(), device="cuda"
        )
    if is_gpu_accelerated:
        model = TSNE_cuml(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            verbose=True,
            metric="cosine",
            method=tsne_method,
        )
        embeddings = np.array(model.fit_transform(embeddings).tolist())
    else:
        model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_jobs=9,
            verbose=True,
            metric="cosine",
            method=tsne_method,
        )
        embeddings = model.fit_transform(embeddings)
    df[[f"{column}_emb_{i}" for i in range(n_components)]] = embeddings
    return df


def target_encoder(
    df: cudf.DataFrame,
    timestamp_col: str,
    grouping_cols: list[str],
    target_col_name: str,
    aggfunc: str,
    lags: list[int],
    dtype: str,
) -> cudf.DataFrame:
    """
    Applies target encoding to the input DataFrame by calculating
    aggregations of the target column based on the provided grouping
    columns and lags.

    Args:
        df (cudf.DataFrame):
            The input DataFrame to apply target encoding to.
        timestamp_col (str):
            The name of the timestamp column in the DataFrame.
        grouping_cols (list[str]):
            A list of column names to group the data by when calculating
            the target encodings.
        target_col_name (str):
            The name of the target column to calculate the encodings for.
        aggfunc (str):
            The aggregation function to use when calculating the target
            encodings (e.g. 'mean', 'median', 'std').
        lags (list[int]):
            A list of lag values to create lagged target encodings for.
        dtype (str):
            The data type to cast the target encoding columns to.

    Returns:
        cudf.DataFrame: The input DataFrame with the added target encoding columns.
    """
    for lag in tqdm(lags, leave=False):

        # calculating encodings
        lag_feat_name = ".".join(
            grouping_cols + [target_col_name] + [aggfunc] + [f"lag.{lag}"]
        )
        encodings = (
            df.groupby([timestamp_col] + grouping_cols)[target_col_name]
            .agg(aggfunc)
            .rename(lag_feat_name)
            .reset_index()
        )
        # shifting timestamp_col columns by lag
        encodings[timestamp_col] += lag

        # Merging encoding with the original data
        df = df.merge(
            encodings, on=[timestamp_col] + grouping_cols, how="left"
        )
        df[lag_feat_name] = df[lag_feat_name].fillna(0)

        # Setting values before the first appearance to nan so they are ignored
        df.loc[df[timestamp_col] <= lag - 1, lag_feat_name] = None

        # for col in grouping_cols:
        #     firsts = df.groupby(col)[timestamp_col].min().rename("firsts")
        #     df = df.merge(firsts, left_on=col, right_index=True, how="left")
        #     df.loc[
        #         df[timestamp_col] < (df["firsts"] + lag), lag_feat_name
        #     ] = None
        #     del df["firsts"]

        # Changing the dtype
        df[lag_feat_name] = df[lag_feat_name].astype(dtype)

    return df
