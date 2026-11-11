"""Load the trained pipeline and score applicant records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd


def predict_one(applicant: dict, model_path: str = "models/credit_scoring_pipeline.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")
    model = joblib.load(path)
    row = pd.DataFrame([applicant])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "probability_of_serious_delinquency": probability,
        "risk_band": (
            "Higher predicted risk" if probability >= 0.50
            else "Lower predicted risk"
        ),
        "threshold": 0.50,
        "disclaimer": "Educational model output; not a lending decision.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to applicant JSON file.")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        applicant = json.load(f)
    print(json.dumps(predict_one(applicant), indent=2))


if __name__ == "__main__":
    main()
