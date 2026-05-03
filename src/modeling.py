"""Model pipelines: baseline, Lasso, and NGBoost implementations.

Pipelines follow sklearn's ColumnTransformer + estimator pattern.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from ngboost import NGBoost
from ngboost.distns import Normal

def build_baseline_pipeline() -> Pipeline:
    # Baseline that predicts the mean
    return Pipeline([("mean", DummyRegressor(strategy="mean"))])


def build_lasso_pipeline(categorical_features):
    # Use dense encoding for simplicity in downstream inspection
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preproc = ColumnTransformer([("cat", cat_pipe, categorical_features)], remainder="drop")
    # Standardize (center) before Lasso so coefficients are comparable
    pipe = Pipeline([("preproc", preproc), ("scaler", StandardScaler(with_mean=True)), ("lasso", Lasso(alpha=0.1, random_state=42, max_iter=5000))])
    return pipe


def build_ngboost_pipeline(categorical_features):
    # NGBoost expects dense arrays; OneHotEncoder produces dense here
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preproc = ColumnTransformer([("cat", cat_pipe, categorical_features)], remainder="drop")
    ngb = NGBoost(Dist=Normal)
    pipe = Pipeline([("preproc", preproc), ("ngboost", ngb)])
    return pipe
