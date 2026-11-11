"""Model explainability helpers (optional SHAP dependency)."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def explain_with_shap(
    data: pd.DataFrame,
    model_path: str = "models/credit_scoring_pipeline.joblib",
    max_rows: int = 200,
):
    """Return SHAP values for a sample.

    TreeExplainer is applied to the fitted Random Forest after preprocessing.
    For a resampling pipeline, SMOTE is skipped automatically at inference.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP with `pip install shap`.") from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("Train the model before requesting explanations.")

    pipeline = joblib.load(path)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    sample = data.head(max_rows)
    transformed = preprocessor.transform(sample)

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(transformed)
    feature_names = preprocessor.get_feature_names_out()
    return explainer, values, feature_names


def main():
    print(
        "Import explain_with_shap(data) from this module and pass a dataframe "
        "containing the same input columns used during training."
    )


if __name__ == "__main__":
    main()
