import gc
import cudf
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
from typing import Tuple

from sales_project.utils import read_yaml, load_model, save_model, save_yaml


class StackingEnsemble:
    """
    Implements a stacking ensemble model.

    The `StackingEnsemble` class is responsible for:
        - Loading and managing the base models and meta-model used in
        the stacking ensemble.
        - Reading the data for first-level and second-level model
        training.
        - Fitting the first-level models and the meta-model.
        - Making predictions using the stacking ensemble.
    """

    def __init__(self, config_path: Path):
        """
        Initializes the StackingEnsemble class with the provided
        configuration.

        The __init__ method reads the configuration from the specified
        YAML file, loads the base models and meta-model, and stores
        them in the self.models dictionary.

        If a fitted model path is provided in the configuration, the
        method will load the pre-trained model. Otherwise, it will create
        a new instance of the model using the parameters specified in the
        YAML file. The meta-model is also loaded or created in a similar
        way.

        Args:
            config_path (Path):
                The path to the configuration YAML file.

        Attributes:
            config_path (Path):
                The path to the configuration YAML file.
            config (dict):
                The configuration dictionary.
            models (dict):
                The dictionary containing the base models and meta-model.
        """

        self.config_path = config_path
        self.config = read_yaml(config_path)

        self.models = {"models": {}, "meta_model": None}
        for model_id, model_params in self.config["models"].items():
            if model_params["fitted_model_path"]:
                self.models["models"][model_id] = load_model(
                    filename=model_params["fitted_model_path"]
                )
            else:
                self.models["models"][model_id] = getattr(
                    __import__(model_params["module"]),
                    model_params["class_name"],
                )(
                    **read_yaml(Path(model_params["params_path"])),
                )
        if self.config["meta_model"]["fitted_model_path"]:
            self.models["meta_model"] = load_model(
                filename=self.config["meta_model"]["fitted_model_path"]
            )
        else:
            self.models["meta_model"] = getattr(
                __import__(self.config["meta_model"]["module"]),
                self.config["meta_model"]["class_name"],
            )(**read_yaml(Path(self.config["meta_model"]["params_path"])))

    def read_data(
        self,
        data_file_path: Path,
        feats: list[str],
        target: str,
        timestamp_col: str,
        timestamps: list[int],
        is_cudf: bool = False,
    ):
        """
        Reads the data from the specified file path, filters
        the data based on the provided timestamps, and returns
        the resulting DataFrame.

        Args:
            data_file_path (Path):
                The path to the data file.
            feats (list[str]):
                The list of feature column names.
            target (str):
                The name of the target column.
            timestamp_col (str):
                The name of the timestamp column.
            timestamps (list[int]):
                The list of timestamps to filter the data by.
            is_cudf (bool, optional):
                Whether to use cuDF instead of pandas.
                Defaults to False.

        Returns:
            pd.DataFrame | cudf.DataFrame:
                The filtered DataFrame.
        """
        if is_cudf:
            df = (
                cudf.read_parquet(
                    data_file_path, columns=feats + [timestamp_col, target]
                )
                .query(f"{timestamp_col} in {timestamps}")
                .drop(timestamp_col, axis=1)
            )
        else:
            df = (
                pd.read_parquet(
                    data_file_path, columns=feats + [timestamp_col, target]
                )
                .query(f"{timestamp_col} in {timestamps}")
                .drop(timestamp_col, axis=1)
            )
        df[target] = df[target].clip(0, 20)
        return df

    def first_level_prediction(
        self,
        df: pd.DataFrame | cudf.DataFrame,
        feats: list[str],
    ) -> Tuple[pd.DataFrame | cudf.DataFrame, list[str]]:
        """
        Generates second-level features by making predictions using
        the first-level models and appending the predictions as new
        features to the input DataFrame.

        Args:
            df (pd.DataFrame | cudf.DataFrame):
                The input DataFrame containing the feature columns.
            feats (list[str]): The list of feature column names.

        Returns:
            pd.DataFrame | cudf.DataFrame, list[str]:
                The updated DataFrame with the second-level features
                added, and the updated list of feature column names.
        """

        is_cudf = isinstance(df, cudf.DataFrame)
        second_level_features = []

        for model_id, model_params in tqdm(self.config["models"].items()):

            prediction_feat = f"{model_params['class_name']}_pred"
            second_level_features.append(prediction_feat)

            if model_params["is_gpu_accelerated"]:
                df[prediction_feat] = (
                    self.models["models"][model_id]
                    .predict(
                        df[feats] if is_cudf else cudf.from_pandas(df[feats])
                    )
                    .clip(0, 20)
                    .values
                )
            else:
                df[prediction_feat] = (
                    self.models["models"][model_id]
                    .predict(df[feats].to_pandas() if is_cudf else df[feats])
                    .clip(0, 20)
                )

        return df, feats + second_level_features

    def fit(
        self,
        data_file_path: Path,
        feats: list[str],
        target: str,
        timestamp_col: str,
        first_level_timestamps: list[int],
        second_level_timestamps: list[int],
        is_cudf: bool = False,
    ):
        """
        Fits the first and second level models of a stacking ensemble
        model.

        Args:
            data_file_path (Path):
                The path to the data file.
            feats (list[str]):
                The list of feature column names.
            target (str):
                The name of the target column.
            timestamp_col (str):
                The name of the timestamp column.
            first_level_timestamps (list[int]):
                The list of timestamps to use for the first level
                models.
            second_level_timestamps (list[int]):
                The list of timestamps to use for the second level
                (meta) model.
            is_cudf (bool, optional):
                Whether the input data is a CuDF DataFrame.
                Defaults to False.

        Returns:
            self:
                The fitted stacking ensemble model.
        """

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if any(
            [v["fitted_model_path"] for v in self.config["models"].values()]
        ):
            print(
                "First level models have already been fitted. "
                "No need to read first level data."
            )
        else:
            print("Reading first level data...")
            df = self.read_data(
                data_file_path=data_file_path,
                feats=feats,
                target=target,
                timestamp_col=timestamp_col,
                timestamps=first_level_timestamps,
                is_cudf=is_cudf,
            )

            # Fitting first level models
            for model_id, model_params in tqdm(self.config["models"].items()):

                if model_params["fitted_model_path"]:
                    print(
                        f"Model {model_params['class_name']} with ID {model_id} "
                        "has already been fitted."
                    )
                else:
                    print(
                        f"Fitting {model_params['class_name']} with ID {model_id}..."
                    )

                    if model_params["is_gpu_accelerated"]:
                        self.models["models"][model_id].fit(
                            (
                                df[feats]
                                if is_cudf
                                else cudf.from_pandas(df[feats])
                            ),
                            (
                                df[target]
                                if is_cudf
                                else cudf.from_pandas(df[target])
                            ),
                        )
                    else:
                        self.models["models"][model_id].fit(
                            df[feats].to_pandas() if is_cudf else df[feats],
                            df[target].to_pandas() if is_cudf else df[target],
                        )

                    filename = f"{model_id}_{model_params['class_name']}_{current_time}.pkl"
                    save_model(
                        model=self.models["models"][model_id],
                        filename=filename,
                    )
                    self.config["models"][model_id][
                        "fitted_model_path"
                    ] = filename
                    save_yaml(path=self.config_path, data=self.config)

            del df
            gc.collect()

        # Second level fitting
        if self.config["meta_model"]["fitted_model_path"]:

            print(
                "Meta model has already been fitted. "
                "No need to read second level data."
            )

        else:

            print("Reading second level data...")
            df = self.read_data(
                data_file_path=data_file_path,
                feats=feats,
                target=target,
                timestamp_col=timestamp_col,
                timestamps=second_level_timestamps,
                is_cudf=is_cudf,
            )

            # Predicting second level data
            print("Predicting using first level models...")
            df, feats2 = self.first_level_prediction(df=df, feats=feats)

            # Fitting meta model
            print(
                f"Fitting meta model {self.config['meta_model']['class_name']}..."
            )
            if self.config["meta_model"]["is_gpu_accelerated"]:
                self.models["meta_model"].fit(
                    (df[feats2] if is_cudf else cudf.from_pandas(df[feats2])),
                    (df[target] if is_cudf else cudf.from_pandas(df[target])),
                )
            else:
                self.models["meta_model"].fit(
                    df[feats2].to_pandas() if is_cudf else df[feats2],
                    df[target].to_pandas() if is_cudf else df[target],
                )
            filename = (
                f"meta_{self.config['meta_model']['class_name']}_"
                f"{current_time}.pkl"
            )
            save_model(
                model=self.models["meta_model"],
                filename=filename,
            )
            self.config["meta_model"]["fitted_model_path"] = filename
            save_yaml(path=self.config_path, data=self.config)

            del df
            gc.collect()

        return self

    def predict(
        self, df: pd.DataFrame | cudf.DataFrame, feats: list[str]
    ) -> pd.DataFrame | cudf.DataFrame:
        """
        Predicts the target variable using the stacked ensemble model.

        Args:
            df (pd.DataFrame | cudf.DataFrame):
                The input data frame containing the features.
            feats (list[str]):
                The list of feature column names.

        Returns:
            pd.DataFrame | cudf.DataFrame:
                The input data frame with the predicted target variable
                added as a new column.
        """

        is_cudf = isinstance(df, cudf.DataFrame)

        # First level prediction
        df, feats2 = self.first_level_prediction(df=df, feats=feats)

        # Second level prediction
        if self.config["meta_model"]["is_gpu_accelerated"]:
            df[f"{self.config['meta_model']['class_name']}_metapred"] = (
                self.models["meta_model"]
                .predict(
                    df[feats2] if is_cudf else cudf.from_pandas(df[feats2])
                )
                .clip(0, 20)
                .values
            )
        else:
            df[f"{self.config['meta_model']['class_name']}_metapred"] = (
                self.models["meta_model"]
                .predict(df[feats2].to_pandas() if is_cudf else df[feats2])
                .clip(0, 20)
            )
        return df
