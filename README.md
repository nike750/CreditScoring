# Credit Scoring & Financial Distress Prediction

An end-to-end machine learning project that predicts the likelihood of serious credit delinquency using historical borrower financial data.

## Project Overview

This project develops a machine learning system to identify borrowers who may experience serious credit delinquency. It uses the Give Me Some Credit dataset and supervised classification techniques.

## Objectives

- Analyze historical borrower financial data.
- Identify patterns associated with credit default.
- Build predictive classification models.
- Handle class imbalance.
- Compare model performance.
- Explain predictions using explainable AI.
- Develop an interactive credit risk assessment application.

## Dataset

Source: Give Me Some Credit – Kaggle.

Dataset: 150,000 borrower records.

Target: `SeriousDlqin2yrs`

- 0: No serious delinquency.
- 1: Serious delinquency.

## Methodology

1. Data collection and exploration.
2. Data cleaning and preprocessing.
3. Exploratory data analysis.
4. Feature engineering.
5. Stratified data splitting.
6. Class imbalance handling.
7. Model training and comparison.
8. Hyperparameter optimization.
9. Model evaluation and calibration.
10. Explainable AI.
11. Model deployment.

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

## Evaluation Metrics

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score
- Confusion Matrix

## Explainable AI

SHAP is used to understand global feature importance and individual prediction explanations.

## Application

The Streamlit application allows users to submit borrower information and obtain a predicted probability of serious delinquency alongside an explanation of the prediction.

## Installation

```bash
git clone https://github.com/nike750/CreditScoring.git
cd CreditScoring
pip install -r requirements.txt
```

## Usage

```bash
python -m src.train
python -m src.evaluate
streamlit run app/streamlit_app.py
```

## Testing

```bash
pytest -v
```

## Limitations and Responsible Use

This project is an educational machine learning exercise. Its predictions are not verified lending decisions or financial advice. The dataset is historical, and model performance does not establish suitability for real-world credit decisions. Fairness, calibration, data drift and external validation require further assessment.

## Future Improvements

- Model monitoring.
- Fairness assessment.
- Automated retraining.
- FastAPI deployment.
- Docker containerization.
- MLflow experiment tracking.

## Author

nike750

## Internship

Developed as part of the Agentic X Machine Learning Internship.
