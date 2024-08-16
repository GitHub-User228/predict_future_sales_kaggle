import gc
import cudf
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import *
from sklearn.model_selection import TimeSeriesSplit

from sales_project.evaluations import evaluate
from sales_project.utils import save_yaml, save_predictions, save_model


def time_series_objective(
    trial: optuna.Trial,
    model_class,
    df: cudf.DataFrame | pd.DataFrame,
    features: list[str],
    target: str,
    scorer: callable,
    hyperparameters_grid: dict,
    cv: int | str = 5,
    val_ratio: float = 0.1,
    timestamp_col: str | None = None,
    step: int = 2,
) -> float:
    """
    Objective function for time series hyperparameter optimization using
    Optuna.

    This function performs hyperparameter optimization for a given model
    and dataset using:
    - TimeSeriesSplit cross-validation
    - Train-Validation split
    - Custom per-timestamp cross-validation.

    Parameters:
        trial (optuna.Trial):
            Optuna trial object for suggesting hyperparameters.
        model_class:
            Class of the machine learning model to be optimized.
        df (cudf.DataFrame | pd.DataFrame):
            DataFrame containing the dataset.
        features (list[str]):
            List of feature column names.
        target (str):
            Target column name.
        scorer (callable):
            Scoring function to evaluate the model's performance.
        hyperparameters_grid (dict):
            Dictionary containing hyperparameter grid configurations.
        cv (int | str, optional):
            Number of splits for TimeSeriesSplit or 'per_timestamp' for custom
            cross-validation. Defaults to 5.
        val_ratio (float, optional):
            Ratio of validation data in the dataset. Defaults to 0.1.
        timestamp_col (str | None, optional):
            Column name for timestamp if using 'per_timestamp' cross-validation.
            Defaults to None.
        step (int, optional):
            Step size for 'per_timestamp' cross-validation. Defaults to 2.

    Returns:
    float: Average score of the model across cross-validation folds.
    """

    hyperparams = {}
    for param, config in hyperparameters_grid.items():
        suggest_method = getattr(trial, config["method"])
        hyperparams[param] = suggest_method(param, **config["kwargs"])

    if type(cv) == int and cv > 1:
        tscv = TimeSeriesSplit(
            n_splits=cv,
            max_train_size=len(df) // cv,
            test_size=int(len(df) // cv * val_ratio),
        )
        scores = []

        for train_index, test_index in tscv.split(df[features + [target]]):
            df_train, df_val = df.iloc[train_index], df.iloc[test_index]

            model = model_class(**hyperparams)
            model.fit(df_train[features], df_train[target])
            y_pred = model.predict(df_val[features]).clip(0, 20)
            del model
            score = scorer(df_val[target], y_pred)
            scores.append(score)
            gc.collect()

        return np.mean(scores)

    elif type(cv) == int and cv == 1:
        split_id = int(len(df) * (1 - val_ratio))
        df_train, df_val = df.iloc[:split_id], df.iloc[split_id:]
        model = model_class(**hyperparams)
        model.fit(df_train[features], df_train[target])
        y_pred = model.predict(df_val[features]).clip(0, 20)
        del model
        gc.collect()
        score = scorer(df_val[target], y_pred)
        return score

    elif type(cv) == str and cv == "per_timestamp":
        scores = []
        for i in range(
            df[timestamp_col].max(), df[timestamp_col].min(), -step
        ):

            model = model_class(**hyperparams)
            model.fit(
                df.query(f"{timestamp_col} < {i}")[features],
                df.query(f"{timestamp_col} < {i}")[target],
            )
            score = scorer(
                df.query(f"{timestamp_col} == {i}")[target],
                model.predict(
                    df.query(f"{timestamp_col} == {i}")[features]
                ).clip(0, 20),
            )
            scores.append(score)
            del model
            gc.collect()

        return np.mean(scores)


def run_optuna(
    df_train: cudf.DataFrame | pd.DataFrame,
    df_test: cudf.DataFrame | pd.DataFrame,
    df_submission: cudf.DataFrame | pd.DataFrame,
    features: list[str],
    target: str,
    model_class,
    objective: callable,
    scorer: callable,
    hyperparameters_grid: dict,
    timestamp_col: str | None = None,
    direction: str = "minimize",
    n_trials: int = 100,
    cv: int = 5,
    val_ratio: float = 0.1,
    step: int = 2,
    is_gpu_accelerated: bool = True,
) -> None:
    """
    Runs Optuna time series hyperparameter optimization

    This function performs hyperparameter optimization for a given model
    and dataset using:
    - TimeSeriesSplit cross-validation
    - Train-Validation split
    - Custom per-timestamp cross-validation.

    Parameters:
        trial (optuna.Trial):
            Optuna trial object for suggesting hyperparameters.
        model_class:
            Class of the machine learning model to be optimized.
        df_train (cudf.DataFrame | pd.DataFrame):
            DataFrame containing the dataset to train on.
        df_test (cudf.DataFrame | pd.DataFrame):
            DataFrame containing the dataset to test on.
        df_submission (cudf.DataFrame | pd.DataFrame):
            DataFrame containing the submission dataset.
        features (list[str]):
            List of feature column names.
        target (str):
            Target column name.
        scorer (callable):
            Scoring function to evaluate the model's performance.
        hyperparameters_grid (dict):
            Dictionary containing hyperparameter grid configurations.
        cv (int | str, optional):
            Number of splits for TimeSeriesSplit or 'per_timestamp' for custom
            cross-validation. Defaults to 5.
        val_ratio (float, optional):
            Ratio of validation data in the dataset. Defaults to 0.1.
        timestamp_col (str | None, optional):
            Column name for timestamp if using 'per_timestamp' cross-validation.
            Defaults to None.
        step (int, optional):
            Step size for 'per_timestamp' cross-validation. Defaults to 2.
        is_gpu_accelerated (bool, optional):
            Whether to use GPU accelerated model training. Defaults to True.
    """
    study = optuna.create_study(direction=direction)
    try:
        study.optimize(
            lambda trial: objective(
                trial=trial,
                model_class=model_class,
                df=df_train,
                features=features,
                target=target,
                timestamp_col=timestamp_col,
                scorer=scorer,
                hyperparameters_grid=hyperparameters_grid,
                cv=cv,
                val_ratio=val_ratio,
                step=step,
            ),
            n_trials=n_trials,
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Exiting optimization earlier.")

    finally:

        # Saving best hyperparams set
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{model_class.__name__}_{current_time}"
        save_yaml(Path(f"../params/best/{filename}.yaml"), study.best_params)

        # Train the model with the best hyperparameters in order to
        # evaluate it on the test set
        print("Fitting the model to evaluate on the test set...")
        best_model = model_class(**study.best_params)
        best_model.fit(df_train[features], df_train[target])
        if is_gpu_accelerated:
            df_test["pred"] = (
                best_model.predict(df_test[features]).clip(0, 20).values
            )
            test_eval = evaluate(df_test.to_pandas(), target, "pred")
        else:
            df_test["pred"] = best_model.predict(df_test[features]).clip(0, 20)
            test_eval = evaluate(df_test, target, "pred")
        print("Test set evaluation: ", test_eval)
        save_yaml(Path(f"../params/best/testeval_{filename}.yaml"), test_eval)
        save_model(best_model, f"test_{filename}.pkl")
        del best_model

        # Train the model with the best hyperparameters in order to
        # make predictions on submission set
        print("Fitting the model to make predictions on submission set...")
        best_model = model_class(**study.best_params)
        if is_gpu_accelerated:
            best_model.fit(
                cudf.concat([df_train[features], df_test[features]]),
                cudf.concat([df_train[target], df_test[target]]),
            )
            df_submission[target] = (
                best_model.predict(df_submission[features]).clip(0, 20).values
            )
            save_predictions(
                df_submission[["shop_id", "item_id", target]].to_pandas(),
                f"{filename}.csv",
            )
        else:
            best_model.fit(
                pd.concat([df_train[features], df_test[features]]),
                pd.concat([df_train[target], df_test[target]]),
            )
            df_submission[target] = best_model.predict(
                df_submission[features]
            ).clip(0, 20)
            save_predictions(
                df_submission[["shop_id", "item_id", target]],
                f"{filename}.csv",
            )
        save_model(best_model, f"{filename}.pkl")
