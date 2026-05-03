"""Feature engineering helpers for categorical-only dataset.

Includes frequency encoding, ordinal size mapping, and simple interactions.
"""
from typing import List
import pandas as pd


def frequency_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)
    return df


def ordinal_encode_size(series: pd.Series) -> pd.Series:
    mapping = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
    return series.map(mapping).fillna(-1).astype(int)


def add_interaction_features(df: pd.DataFrame, pairs: List[tuple]) -> pd.DataFrame:
    df = df.copy()
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}__{b}"] = df[a].astype(str) + "__" + df[b].astype(str)
    return df
