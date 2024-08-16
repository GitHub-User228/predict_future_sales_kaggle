import re
import cudf
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def read_yaml(path: Path) -> dict:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path_to_yaml (Path):
            Path to the yaml file

    Returns:
        Dict:
            The yaml content as a dict.

    Raises:
        ValueError:
            If the file is not a YAML file
        FileNotFoundError:
            If the file is not found.
        yaml.YAMLError:
            If there is an error parsing the yaml file.
    """
    if path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"The file {path} is not a YAML file")
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
        return content
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while reading YAML file: {e}"
        )


def save_yaml(path: Path, data: dict):
    """
    Save yaml data

    Args:
        path (Path): path to yaml file
        data (dict): data to be saved in yaml file
    """
    try:
        with open(path, "w") as f:
            yaml.dump(data, f, indent=4)
        print(f"yaml file saved at: {path}")
    except PermissionError:
        print(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        print(f"Failed to save yaml to {path}. Error: {e}")
        raise


def clean_string(string: str) -> str:
    """
    Cleans a string by removing bracketed terms, special characters,
    and extra whitespace, and converts the string to lowercase.

    Args:
        string (str):
            The input string to be cleaned.

    Returns:
        str:
            The cleaned string.
    """
    string = re.sub(r"\[.*?\]", "", string)
    string = re.sub(r"\(.*?\)", "", string)
    string = re.sub(r"[^A-ZА-Яa-zа-я0-9 ]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string


def iqr_filter(df: pd.DataFrame, feature: str, k: float = 1.5) -> None:
    """
    Filters a DataFrame by removing rows with feature values outside
    the interquartile range (IQR).

    Args:
        df (pd.DataFrame):
            The input DataFrame to be filtered.
        feature (str):
            The name of the feature column to filter on.
        k (float, optional):
            The multiplier for the IQR to determine the lower and upper
            bounds. Defaults to 1.5.
    """
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    original_length = len(df)

    df.drop(
        df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index,
        axis=0,
        inplace=True,
    )
    print(f"Initial number of observations: {original_length}")
    print(
        f'Number of observations after IQR filtering along feature "{feature}": {len(df)}'
    )
    print(f"Number of observations dropped: {original_length - len(df)}")
    print(f"Drop rate: {1-len(df)/original_length}")


def add_lag(
    df: cudf.DataFrame | pd.DataFrame,
    lags: list[int],
    feat_to_lag: str,
    timestamp_col: str,
    grouping_cols: list[str],
    dtype: str,
) -> cudf.DataFrame | pd.DataFrame:
    """
    Adds lagged features to a DataFrame.

    Args:
        df (cudf.DataFrame | pd.DataFrame):
            The input DataFrame to add lagged features to.
        lags (list[int]):
            A list of lag values to create lagged features for.
        feat_to_lag (str):
            The name of the feature column to create lagged features
            for.
        timestamp_col (str):
            The name of the timestamp column in the DataFrame.
        grouping_cols (list[str]):
            A list of column names to group the data by when creating
            lagged features.
        dtype (str):
            The data type to cast the lagged feature columns to.

    Returns:
        cudf.DataFrame | pd.DataFrame:
            The input DataFrame with the added lagged features.
    """
    for lag in tqdm(lags):
        lag_feat_name = f"{feat_to_lag}.lag.{lag}"
        lagged = df.loc[:, [timestamp_col] + grouping_cols + [feat_to_lag]]
        lagged.rename(columns={feat_to_lag: lag_feat_name}, inplace=True)
        lagged[timestamp_col] += lag
        df = df.merge(lagged, on=[timestamp_col] + grouping_cols, how="left")
        df[lag_feat_name] = df[lag_feat_name].fillna(0)
        df.loc[df[timestamp_col] <= lag - 1, lag_feat_name] = None
        df[lag_feat_name] = df[lag_feat_name].astype(dtype)
    return df


def get_bins(x: int) -> int:
    """
    Calculates the appropriate number of bins for the histogram
    according to the number of the observations

    Args:
        x (int):
            Number of the observations

    Returns:
        int:
            Number of bins
    """
    if x > 0:
        n_bins = max(int(1 + 3.2 * np.log(x)), int(1.72 * x ** (1 / 3)))
    else:
        message = (
            "An invalid input value passed. Expected a positive "
            + "integer, but got {x}"
        )
        raise ValueError(message)
    return n_bins


def save_predictions(df_submission, filename: str) -> None:
    """
    Saves the predicted item counts for the test set to a CSV file.

    Args:
        df_submission (pd.DataFrame):
            A DataFrame containing the predicted item counts
            for each shop and item.
        filename (str):
            The name of the csv file to save the predictions to
    """
    submission = pd.read_csv("../data/raw/test.csv")
    submission = submission.merge(
        df_submission[["shop_id", "item_id", "item_cnt_month"]],
        on=["shop_id", "item_id"],
        how="left",
    )
    submission[["ID", "item_cnt_month"]].to_csv(
        f"../data/predictions/{filename}", index=False
    )
    print(f"csv file saved at: ../data/predictions/{filename}")


def reduce_size(df):
    """
    Reduces the size of the DataFrame by converting integer
    and float columns to smaller data types.

    This function iterates through each column in the DataFrame and
    checks the minimum and maximum values. It then converts the column
    to a smaller data type if possible, such as `uint8`, `uint16`,
    `int8`, or `int16`, to reduce the memory footprint of the DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame to be reduced in size.
    """
    for col in tqdm(df.columns):
        if "int" in df[col].dtype.name:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if max(abs(df[col].min()), df[col].max()) <= 127:
                    df[col] = df[col].astype("int8")
                elif max(abs(df[col].min()), df[col].max()) <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")
        elif "float" in df[col].dtype.name:
            df[col] = df[col].astype("float32")


def save_model(model, filename: str) -> None:
    """
    Saves the model to a binary file.
    """
    with open(Path(f"../models/{filename}"), "wb") as f:
        pickle.dump(model, f)
    print(f"pkl file saved at: ../models/{filename}")


def load_model(filename: str):
    """
    Loads the model from a binary file.
    """
    with open(Path(f"../models/{filename}"), "rb") as f:
        model = pickle.load(f)
    return model
