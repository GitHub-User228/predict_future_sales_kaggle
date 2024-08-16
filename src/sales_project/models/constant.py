import pandas as pd
from tqdm.auto import tqdm

from sales_project.evaluations import METRICS, evaluate


class ConstantModel:

    def fit_predict(
        self, df: pd.DataFrame, target_col: str, group: list[str] = []
    ):

        if len(group) > 0:
            df["prediction"] = df.groupby(group)[target_col].shift()
        else:
            df["prediction"] = df[target_col].shift()
        return df

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: str,
        metrics: list[METRICS] = ["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"],
        groups: list[str] = [],
    ):
        metrics_values = pd.DataFrame(columns=metrics + ["averaging", "group"])
        it = 0
        for group in tqdm(groups):
            for name in ("median", "mean"):
                df2 = (
                    df[[timestamp_col, target_col] + group]
                    .groupby([timestamp_col] + group)
                    .agg({target_col: name})
                    .reset_index()
                )
                df2 = self.fit_predict(
                    df=df2, target_col=target_col, group=group
                )
                df2 = df2.drop(target_col, axis=1)
                df2 = df.merge(
                    df2, on=[timestamp_col] + group, how="left"
                ).dropna(axis=0)
                values = evaluate(
                    data=df2,
                    target_column=target_col,
                    prediction_column="prediction",
                    metrics=metrics,
                )
                metrics_values.loc[it] = values
                metrics_values.loc[it, "averaging"] = name
                metrics_values.loc[it, "group"] = f"{group}"
                it += 1
        return metrics_values
