from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import *


METRICS = Literal["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"]


def smape(y_true, y_pred) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)
    between the true and predicted values.

    Args:
        y_true (array-like):
            The true target values.
        y_pred (array-like):
            The predicted target values.

    Returns:
        float:
            The SMAPE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff) * 100


METRICS_FUNCTIONS = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "R2": r2_score,
    "MAPE": mean_absolute_percentage_error,
    "SMAPE": smape,
}


def evaluate(
    data: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    metrics: list[METRICS] = ["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"],
) -> dict:
    """
    Evaluates the performance of a machine learning model by calculating
    various metrics on the actual and predicted target values.

    Args:
        data (pd.DataFrame):
            The input data containing the actual and predicted target
            values.
        target_column (str):
            The name of the column in `data` containing the actual
            target values.
        prediction_column (str):
            The name of the column in `data` containing the predicted
            target values.
        metrics (list[METRICS]):
            The list of metrics to calculate.

    Returns:
        dict: A dictionary containing some of the following performance
            metrics:
            - MAE: Mean Absolute Error
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - R2: Coefficient of Determination
            - MAPE: Mean Absolute Percentage Error
            - SMAPE: Symmetric Mean Absolute Percentage Error
    """
    values = {}
    for metric in metrics:
        if metric == "RMSE":
            if "MSE" in values:
                values[metric] = float(values["MSE"] ** (0.5))
            else:
                values[metric] = float(
                    METRICS_FUNCTIONS["MSE"](
                        data[target_column], data[prediction_column]
                    )
                    ** (0.5)
                )
        else:
            values[metric] = float(
                METRICS_FUNCTIONS[metric](
                    data[target_column], data[prediction_column]
                )
            )
    return values
