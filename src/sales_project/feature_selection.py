import cudf
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from sales_project.evaluations import evaluate, METRICS


K = {"MAE": 1, "MSE": 1, "RMSE": 1, "R2": -1, "MAPE": 1, "SMAPE": 1}


class PermutationImportance:
    """
    Permutation Importance Evaluator for a Machine Learning Model.

    This class provides a way to evaluate the importance of features
    in a machine learning model by calculating the change in model
    performance metrics when each feature is randomly permuted.
    """

    def __init__(
        self,
        model: object,
        features: list[str],
        target: str,
        n_permutations: int,
        metrics: list[METRICS] = ["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"],
        is_gpu_accelerated: bool = True,
    ) -> None:
        """
        Initializes a PermutationImportanceGPU object with the
        provided parameters.

        Args:
            model (object):
                The machine learning model to evaluate (not fitted).
            features (list[str]):
                The list of feature names to use to fit the model.
            target (str):
                The name of the target variable.
            n_permutations (int):
                The number of permutations to perform for each feature.
            metrics (list[METRICS], optional):
                The list of evaluation metrics to use.
                Defaults to ["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE"].
            is_gpu_accelerated (bool, optional):
                Whether the model is GPU accelerated.
        """
        self.model = model
        self.features = features
        self.target = target
        self.n_permutations = n_permutations
        self.metrics = metrics
        self.is_gpu_accelerated = is_gpu_accelerated

    def predict_and_evaluate(self, df: cudf.DataFrame):
        """
        Predicts on the validation set and calculates the specified
        evaluation metrics.

        Args:
            df (cudf.DataFrame):
                The DataFrame containing the validation data.

        Returns:
            pd.DataFrame:
                A DataFrame containing the calculated metric values.
        """
        if self.is_gpu_accelerated:
            df["pred"] = self.model.predict(df[self.features]).values
        else:
            df["pred"] = self.model.predict(df[self.features])
        data = df[[self.target, "pred"]]
        if self.is_gpu_accelerated:
            data = data.to_pandas()
        metrics_values = evaluate(
            data=data,
            target_column=self.target,
            prediction_column="pred",
            metrics=self.metrics,
        )
        return metrics_values

    def evaluate_feature(
        self,
        feature_to_permutate: str,
        baseline_metrics: pd.DataFrame,
        df: cudf.DataFrame,
    ):
        """
        Evaluates the importance of a single feature by permuting its
        values and calculating the change in evaluation metrics
        compared to the baseline.

        Args:
            feature_to_permutate (str | None):
                The name of the feature to permute.
            baseline_metrics (pd.DataFrame):
                The baseline evaluation metrics calculated on the
                original data.
            df (cudf.DataFrame):
                The DataFrame containing the validation data.

        Returns:
            pd.DataFrame:
                A DataFrame containing the calculated importance values
                for each evaluation metric.
        """

        df.rename(
            columns={feature_to_permutate: f"{feature_to_permutate}__"},
            inplace=True,
        )

        metrics_values = pd.DataFrame(columns=self.metrics)

        for it in range(self.n_permutations):

            # permutatiting
            shuffled = (
                df[f"{feature_to_permutate}__"]
                .sample(frac=1.0)
                .reset_index(drop=True)
            ).values
            df.loc[:, feature_to_permutate] = shuffled

            metrics_values.loc[it] = self.predict_and_evaluate(df=df)

        # Dropping the permutated column and renaming the original one
        del df[feature_to_permutate]
        df.rename(
            columns={f"{feature_to_permutate}__": feature_to_permutate},
            inplace=True,
        )

        # Calculating importance based on the baseline
        importances = baseline_metrics.values() - metrics_values

        return importances

    def evaluate_features(
        self,
        features_to_permutate: list[str],
        df: pd.DataFrame | cudf.DataFrame,
    ):
        """
        Evaluates the importance of a set of features by permuting
        their values and calculating the change in evaluation
        metrics compared to the baseline.
        Also save the importance values in a csv file.

        Args:
            features_to_permutate (list[str]):
                The names of the features to permute.
            df (pd.DataFrame | cudf.DataFrame):
                The DataFrame containing the validation data.
            imp_filename (str):
                The name of the csv file to save the importance values.

        Returns:
            pd.DataFrame:
                A DataFrame containing the calculated importance
                values for each evaluation metric and feature.
        """

        baseline_metrics = pd.DataFrame(columns=self.metrics)
        baseline_metrics = self.predict_and_evaluate(df=df)

        importances = pd.DataFrame(columns=self.metrics)

        iterator = tqdm(
            enumerate(features_to_permutate),
            total=len(features_to_permutate),
            desc=f"Evaluating importance of features",
        )
        iterator.set_postfix({"feature": None})

        # Evaliating importance of each feature
        for it, feature in iterator:
            iterator.set_postfix({"feature": feature})
            importances__ = self.evaluate_feature(
                feature_to_permutate=feature,
                baseline_metrics=baseline_metrics,
                df=df,
            )
            importances__["index"] = [
                f"{feature}_{k}" for k in range(self.n_permutations)
            ]
            importances__.set_index("index", inplace=True)
            importances = pd.concat([importances, importances__], axis=0)

        for metric in self.metrics:
            importances[metric] *= K[metric]
        importances.reset_index(inplace=True)
        importances["group"] = importances["index"].apply(
            lambda x: int(x.split("_")[-1])
        )
        importances["feature"] = importances["index"].apply(
            lambda x: "_".join(x.split("_")[:-1])
        )
        importances.drop("index", axis=1, inplace=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{self.model.__class__.__name__}_{current_time}"
        importances.to_csv(f"../data/importances/{filename}.csv", index=False)

        return importances


class ImportanceBasedBestFeatureSetSearch:
    """
    Importance-based best feature set search class.

    This class provides a way to select the best subset of features
    for a given machine learning model and improtance values of
    each feature.
    """

    def __init__(
        self,
        model_class,
        model_params: dict,
        features_df: pd.DataFrame,
        features_col: str,
        timestamp_col: str,
        target: str,
        val_ratio: float,
        cv: int | str,
        cv_step: int,
        metric_name: str,
        scorer: callable,
    ):
        """
        Initializes an instance of the `ImportanceBasedBestFeatureSetSearch` class.

        Args:
            model_class (callable):
                The class of the machine learning model to use.
            model_params (dict):
                The parameters to pass to the model class when initializing the model.
            features_df (pd.DataFrame):
                The DataFrame containing info about importances of features .
            features_col (str):
                The name of the column in `features_df` that contains the feature names.
            timestamp_col (str):
                The name of the column in the input DataFrame that contains the timestamp.
            target (str):
                The name of the target variable to predict.
            val_ratio (float):
                The ratio of the dataset to use for validation.
            cv (int | str):
                The number of cross-validation folds to use,
                or "per_timestamp" to use a time-series split.
            cv_step (int):
                The step size for the time-series cross-validation.
            metric_name (str):
                The name of the evaluation metric to use.
            scorer (callable):
                The function to use for scoring the model's predictions.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.features_df = features_df
        self.features_col = features_col
        self.timestamp_col = timestamp_col
        self.target = target
        self.scorer = scorer
        self.val_ratio = val_ratio
        self.cv = cv
        self.cv_step = cv_step
        self.metric_name = metric_name

    def evaluate(
        self, df: pd.DataFrame | cudf.DataFrame, features: list[str]
    ) -> float:
        """
        Evaluates the performance of a machine learning model
        using the specified features and cross-validation strategy.

        Args:
            df (pd.DataFrame | cudf.DataFrame):
                The input DataFrame containing the data.
            features (list[str]):
                The list of feature names to use for the model.

        Returns:
            float:
                The average score of the model across
                the cross-validation folds.
        """

        if type(self.cv) == int and self.cv > 1:
            tscv = TimeSeriesSplit(
                n_splits=self.cv,
                max_train_size=len(df) // self.cv,
                test_size=int(len(df) // self.cv * self.val_ratio),
            )
            scores = []

            for train_index, test_index in tscv.split(
                df[features + [self.target]]
            ):
                df_train, df_val = df.iloc[train_index], df.iloc[test_index]

                model = self.model_class(**self.model_params)
                model.fit(df_train[features], df_train[self.target])
                y_pred = model.predict(df_val[features])
                try:
                    y_pred = y_pred.clip(lower=0)
                except:
                    y_pred = y_pred.clip(min=0)
                score = self.scorer(df_val[self.target], y_pred)
                scores.append(score)

            return np.mean(scores)

        elif type(self.cv) == int and self.cv == 1:
            split_id = int(len(df) * (1 - self.val_ratio))
            df_train, df_val = df.iloc[:split_id], df.iloc[split_id:]
            model = self.model_class(**self.model_params)
            model.fit(df_train[features], df_train[self.target])
            y_pred = model.predict(df_val[features])
            try:
                y_pred = y_pred.clip(lower=0)
            except:
                y_pred = y_pred.clip(min=0)
            score = self.scorer(df_val[self.target], y_pred)
            return score

        elif type(self.cv) == str and self.cv == "per_timestamp":
            scores = []
            for i in range(
                df[self.timestamp_col].max(),
                df[self.timestamp_col].min(),
                -self.cv_step,
            ):

                model = self.model_class(**self.model_params)
                model.fit(
                    df.query(f"{self.timestamp_col} < {i}")[features],
                    df.query(f"{self.timestamp_col} < {i}")[self.target],
                )
                y_pred = model.predict(
                    df.query(f"{self.timestamp_col} == {i}")[features]
                )
                try:
                    y_pred = y_pred.clip(lower=0)
                except:
                    y_pred = y_pred.clip(min=0)
                score = self.scorer(
                    df.query(f"{self.timestamp_col} == {i}")[self.target],
                    y_pred,
                )
                scores.append(score)

            return np.mean(scores)

    def fit(
        self,
        df: pd.DataFrame | cudf.DataFrame,
        min_feature_id: int,
        max_feature_id: int,
        step: int,
    ) -> pd.DataFrame:
        """
        Evaluates the performance of a model using a
        specified number of features based on their
        importances.

        Args:
            df (pd.DataFrame | cudf.DataFrame):
                The input DataFrame containing the features and
                target variable.
            min_feature_id (int):
                The minimum number of features.
            max_feature_id (int):
                The maximum number of features.
            step (int):
                The step size for the number of features.

        Returns:
            pd.DataFrame:
                A dataframe containing the number of features and
                the corresponding metric values.
        """

        metrics = {"n_features": [], self.metric_name: []}

        iterator = tqdm(range(min_feature_id, max_feature_id, step))
        iterator.set_postfix({"n_features": None})

        try:

            for n_features in iterator:
                iterator.set_postfix({"n_features": n_features})

                features = list(
                    self.features_df.iloc[:n_features][self.features_col]
                )
                value = self.evaluate(df=df, features=features)
                metrics["n_features"].append(n_features)
                metrics[self.metric_name].append(value)
                print(
                    f"Number of features: {n_features}, {self.metric_name}: {value}"
                )

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Exiting earlier.")

        finally:
            metrics = pd.DataFrame(metrics)

            current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filename = f"{self.model_class.__name__}_{current_time}"
            metrics.to_csv(
                f"../data/feature_selection/{filename}.csv", index=False
            )

            return metrics
