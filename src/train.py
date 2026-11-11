"""Train and persist a leakage-conscious credit-risk baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from .data_preprocessing import load_data, make_preprocessor, prepare_xy


def build_pipeline(X, use_smote: bool = True):
    preprocessor = make_preprocessor(X)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None if use_smote else "balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            return ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ])
        except ImportError:
            print("imbalanced-learn unavailable; using class_weight='balanced_subsample'.")
            model.set_params(class_weight="balanced_subsample")

    return SklearnPipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cs-training.csv")
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()

    data = load_data(args.data)
    X, y = prepare_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline = build_pipeline(X_train, use_smote=not args.no_smote)
    pipeline.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/credit_scoring_pipeline.joblib")

    metadata = {
        "model": "RandomForestClassifier",
        "target": "SeriousDlqin2yrs",
        "random_state": 42,
        "test_size": 0.20,
        "features": X.columns.tolist(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "smote_enabled": not args.no_smote,
        "note": "Test set held out; use evaluate.py for final test metrics.",
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Keep the held-out split available to evaluate.py without fitting on it.
    test = X_test.copy()
    test["SeriousDlqin2yrs"] = y_test
    test.to_csv("reports/test_split.csv", index=False)
    print("Saved model to models/credit_scoring_pipeline.joblib")


if __name__ == "__main__":
    main()
