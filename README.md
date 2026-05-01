# Employee Attrition Prediction — End-to-End DS Project

## Project Overview
An end-to-end Data Science project that predicts employee attrition using ML and NLP, with a live interactive dashboard.

## Business Problem
HR teams lose significant time and money to unexpected attrition. This project answers two questions:
- **Who** is likely to leave? (ML model)
- **Why** are they leaving? (NLP on exit surveys)

## Key Findings
- Overall attrition rate: **16.1%**
- Overtime is the single strongest predictor — employees doing overtime leave at **3x the rate**
- Sales Representatives have the highest attrition at **~40%**
- Environment dissatisfaction carries the most negative exit sentiment (-0.45 avg score)
- Both ML and NLP analyses independently point to the same root causes

## Tech Stack
- **Data:** IBM HR Analytics Dataset (1,470 employees, 35 features)
- **ML Models:** Logistic Regression, Random Forest, XGBoost
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **NLP:** VADER Sentiment Analysis, LDA Topic Modeling
- **Dashboard:** Streamlit
- **Imbalance Handling:** SMOTE

## Model Performance
| Model | ROC-AUC (Test) | CV AUC |
|---|---|---|
| Logistic Regression | 0.802 | 0.859 |
| XGBoost | 0.796 | 0.981 |
| Random Forest | 0.794 | 0.987 |

Logistic Regression selected as primary model due to superior recall on the minority class (37/47 at-risk employees correctly identified).

## Project Structure