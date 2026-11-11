"""Data loading, validation, splitting, and preprocessing."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import TARGET, add_features, split_features_target

DEFAULT_DATA_PATH = Path("data/raw/cs-training.csv")


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download cs-training.csv from Kaggle "
            "and place it in data/raw/."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"{path} is empty or contains only a header. Replace the placeholder "
            "with the actual Kaggle training dataset."
        )
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    # Kaggle's exported training file sometimes includes an unnamed index column.
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def prepare_xy(df: pd.DataFrame):
    df = add_features(df)
    return split_features_target(df)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted only inside the training pipeline."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = SklearnPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("numeric", numeric_pipeline, numeric_features)]

    if categorical_features:
        categorical_pipeline = SklearnPipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
