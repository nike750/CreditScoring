# 💳 Credit Scoring & Financial Distress Prediction Pipeline

> An end-to-end machine learning pipeline built to evaluate loan applicant creditworthiness and predict financial distress using historical financial data. Developed to fulfill the requirements of the Agentic X machine learning internship.

## 🛠️ Tools & Tech Stack
* **Language & Core Libraries:** Python, Pandas, NumPy, Scikit-Learn
* **Imbalanced Learning:** `imbalanced-learn` (SMOTE)
* **Model Training & Ensembles:** Scikit-Learn (Logistic Regression, Random Forest Classifier)
* **Evaluation & Metrics:** Classification Reports, ROC-AUC, Precision-Recall AUC
* **Environment:** Google Colab / Jupyter Notebooks

## 📊 Dataset & Features Description
The project utilizes the official **"Give Me Some Credit"** Kaggle dataset (`cs-training.csv`), containing **150,000 rows** of historical borrower records.

### Core Features:
1. **`SeriousDlqin2yrs` (Target):** Person experienced 90 days past due delinquency or worse (Binary: 1 = Default/Financial Distress, 0 = Safe).
2. **`RevolvingUtilizationOfUnsecuredLines`:** Total balance on credit cards and personal lines of credit except real estate and no installment debt divided by the sum of credit limits.
3. **`age`:** Age of borrower in years.
4. **`NumberOfTime30-59DaysPastDueNotWorse`:** Number of times borrower has been 30-59 days past due.
5. **`DebtRatio`:** Monthly debt payments, alimony, and living costs divided by monthly gross income.
6. **`MonthlyIncome`:** Monthly income of the borrower.
7. **`NumberOfOpenCreditLinesAndLoans`:** Number of open loans (installment and revolving) and lines of credit.
8. **`NumberOfTimes90DaysLate`:** Number of times borrower has been 90 days past due or more.
9. **`NumberRealEstateLoansOrLines`:** Number of mortgage and real estate loans including home equity lines of credit.
10. **`NumberOfTime60-89DaysPastDueNotWorse`:** Number of times borrower has been 60-89 days past due.
11. **`NumberOfDependents`:** Number of dependents in family excluding themselves.

## 🔄 Project Workflow

```
[Data Ingestion (150k rows)] 
       │
       ▼
[Data Cleaning & Imputation (Median filling for Income & Dependents)]
       │
       ▼
[Feature Engineering (Credit ratios & aggregate delinquency indicators)]
       │
       ▼
[Stratified Train-Test Split (80/20 ratio maintaining class proportions)]
       │
       ▼
[Class Imbalance Mitigation via SMOTE (Applied ONLY to training data)]
       │
       ▼
[Model Training & Benchmarking (Logistic Regression vs. Random Forest)]
       │
       ▼
[Evaluation & Interpretation (ROC-AUC, Recall optimization & Feature Importance)]
```

## 🤖 Models Evaluated & Performance

Models were strictly evaluated using **Precision, Recall, F1-Score, and ROC-AUC**, avoiding misleading accuracy metrics on imbalanced data.

| Model | ROC-AUC Score | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.7127 | 0.15 | 0.50 | 0.24 |
| **Tree Ensemble (Random Forest)** | **0.8527** | **0.22** | **0.72** | **0.34** |

### 🏆 Best Model: Random Forest Tree Ensemble
* **Why it wins:** It achieved a strong ROC-AUC of **0.8527** and a high default **Recall of 72%**. In a banking context, missing a high-risk defaulter causes massive direct capital loss, making high recall vital.

## ⚠️ Limitations & Future Improvements

1. **Class Overlap & False Positives:** While SMOTE successfully raised the recall to 72%, it increased false positives (precision dropped to 22%), meaning some safe credit applicants are flagged as risky. Fine-tuning decision probability thresholds via cost-sensitive learning can optimize this balance.
2. **Missing Data Imputation:** Median imputation was used for missing `MonthlyIncome` and `NumberOfDependents`. Implementing advanced iterative imputers (like KNN Imputer or MICE) could preserve data variance better.
3. **Hyperparameter Tuning:** The models used default or standard tree depths. Incorporating `GridSearchCV` or `Optuna` would extract peak performance from the ensemble.