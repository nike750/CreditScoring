"""Evaluate a saved model on the held-out test split."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report,
)

TARGET = "SeriousDlqin2yrs"


def main():
    model_path = Path("models/credit_scoring_pipeline.joblib")
    test_path = Path("reports/test_split.csv")
    if not model_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python -m src.train` first.")

    model = joblib.load(model_path)
    test = pd.read_csv(test_path)
    y_test = test.pop(TARGET).astype(int)
    probabilities = model.predict_proba(test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
        "threshold": 0.50,
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("classification_report",)}, indent=2))


if __name__ == "__main__":
    main()
