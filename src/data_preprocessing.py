"""Data loading and basic preprocessing utilities.

Keep this module small: convert categorical columns to `category` dtype
and provide a reproducible train/test split.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str, dtype_map: dict = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if dtype_map:
        df = df.astype(dtype_map)
    # ensure categorical columns are category dtype
    for col in ["Brand", "Category", "Color", "Size", "Material"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def split_data(df: pd.DataFrame, target: str = "Price", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
