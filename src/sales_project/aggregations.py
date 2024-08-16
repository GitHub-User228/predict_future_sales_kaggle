import cudf


def rolling(
    df: cudf.DataFrame,
    timestamp_feat: str,
    grouping_feats: list[str],
    feat_to_agg: str,
    new_feat_name: str,
    aggfunc: str,
    window: int,
    dtype: str,
    lag_offset=0,
) -> cudf.DataFrame:
    """
    Calculates lagged rolling aggregations for a given DataFrame,
    timestamp feature, grouping features, and aggregation function.

    Args:
        df (cudf.DataFrame):
            The input DataFrame.
        timestamp_feat (str):
            The name of the timestamp feature column.
        grouping_feats (list[str]):
            A list of feature names to group by.
        feat_to_agg (str):
            The name of the feature to aggregate.
        new_feat_name (str):
            The name of the new feature to create with the
            rolling aggregation.
        aggfunc (str):
            The aggregation function to apply
            (e.g. 'mean', 'sum', 'min', 'max').
        window (int):
            The size of the rolling window.
        dtype (str):
            The data type of the new feature.
        lag_offset (int, optional):
            The number of periods to offset the rolling window by.
            Defaults to 0.

    Returns:
        cudf.DataFrame:
            A DataFrame with the rolling aggregations.
    """
    aggregations = []
    for i in range(2, df[timestamp_feat].max() + 1):
        aggregation = (
            df[
                (df[timestamp_feat] >= max([i - window - lag_offset, 0]))
                & (df[timestamp_feat] < i - lag_offset)
            ]
            .groupby(grouping_feats)[feat_to_agg]
            .agg(aggfunc)
            .astype(dtype)
            .rename(new_feat_name)
            .reset_index()
        )
        aggregation[timestamp_feat] = i
        aggregations.append(aggregation)
    aggregations = cudf.concat(aggregations, axis=0)
    return aggregations


def aggregate(
    df: cudf.DataFrame,
    timestamp_feat: str,
    grouping_feats: list[str],
    feat_to_agg: str,
    reshaping_aggfunc: str,
    rolling_aggfunc: str,
    window: int,
    dtype: str,
    lag_offset=0,
    requires_reshaping: bool = False,
    kind: str = "rolling",
):
    """
    Applies a rolling or expanding aggregation to a DataFrame,
    with optional reshaping.

    Args:
        df (cudf.DataFrame):
            The input DataFrame.
        timestamp_feat (str):
            The name of the timestamp feature column.
        grouping_feats (list[str]):
            A list of feature names to group by.
        feat_to_agg (str):
            The name of the feature to aggregate.
        reshaping_aggfunc (str):
            The aggregation function to use for reshaping the DataFrame.
        rolling_aggfunc (str):
            The aggregation function to use for the rolling/expanding window.
        window (int):
            The size of the rolling window.
        dtype (str):
            The data type of the new feature.
        lag_offset (int, optional):
            The number of periods to offset the rolling window by.
            Defaults to 0.
        requires_reshaping (bool, optional):
            Whether the DataFrame requires reshaping before applying
            the rolling/expanding aggregation. Defaults to False.
        kind (str, optional):
            The type of aggregation to perform, either "rolling" or
            "expanding". Defaults to "rolling".

    Returns:
        cudf.DataFrame:
            A DataFrame with the rolling/expanding aggregations.
    """
    if requires_reshaping:
        # Forming a table for all possible combinations of the grouping
        # features and timestamp_feat feature
        # In case of not existing combination, feat_to_agg value is set
        # to 0
        df2 = (
            df[[timestamp_feat, feat_to_agg] + grouping_feats]
            .to_pandas()
            .pivot_table(
                index=[timestamp_feat] + grouping_feats,
                values=feat_to_agg,
                aggfunc=reshaping_aggfunc,
                dropna=False,
                fill_value=0,
            )
            .reset_index()
        )
        df2 = cudf.from_pandas(df2)
        # Set values before the first appearance to nan so they
        # are ignored rather than being treated as zero sales.
        for feat in grouping_feats:
            firsts = df.groupby(feat)[timestamp_feat].min().rename("firsts")
            df2 = df2.merge(firsts, left_on=feat, right_index=True, how="left")
            df2.loc[df2[timestamp_feat] < df2["firsts"], feat_to_agg] = None
            del df2["firsts"]
        df2.reset_index(inplace=True)
    else:
        df2 = df[[timestamp_feat, feat_to_agg] + grouping_feats]

    if kind == "rolling":
        feat_name = (
            f"{'.'.join(grouping_feats)}"
            f".{feat_to_agg}"
            f".{reshaping_aggfunc}"
            f".rolling.{rolling_aggfunc}"
            f".win.{window}"
        )
        print(f'Creating feature "{feat_name}"')
        return (
            rolling(
                df=df2,
                timestamp_feat=timestamp_feat,
                grouping_feats=grouping_feats,
                feat_to_agg=feat_to_agg,
                new_feat_name=feat_name,
                aggfunc=rolling_aggfunc,
                window=window,
                dtype=dtype,
                lag_offset=lag_offset,
            ),
            feat_name,
        )
    elif kind == "expanding":
        feat_name = (
            f"{'.'.join(grouping_feats)}"
            f".{feat_to_agg}"
            f".{reshaping_aggfunc}"
            f".expanding.{rolling_aggfunc}"
        )
        print(f'Creating feature "{feat_name}"')
        return (
            rolling(
                df=df2,
                timestamp_feat=timestamp_feat,
                grouping_feats=grouping_feats,
                feat_to_agg=feat_to_agg,
                new_feat_name=feat_name,
                aggfunc=rolling_aggfunc,
                window=100,
                dtype=dtype,
                lag_offset=lag_offset,
            ),
            feat_name,
        )
    elif kind == "ewm":
        feat_name = (
            f"{'.'.join(grouping_feats)}"
            f".{feat_to_agg}"
            f".{reshaping_aggfunc}"
            f".ewm.{rolling_aggfunc}"
        )
        print(f'Creating feature "{feat_name}"')
        df2 = df2.to_pandas()
        df2[feat_name] = (
            df2.groupby(grouping_feats)[feat_to_agg]
            .ewm(halflife=window, min_periods=1)
            .agg(rolling_aggfunc)
            .to_numpy(dtype=dtype)
        )
        df2[timestamp_feat] += 1 - lag_offset
        del df2[feat_to_agg]
        df2 = cudf.from_pandas(df2).drop("index", axis=1)
        return df2, feat_name
