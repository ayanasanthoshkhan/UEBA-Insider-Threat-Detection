# UEBA Insider Threat Detection

Big Data-based User Behavior Analytics for insider threat detection using Apache Spark, Machine Learning, and the CERT r4.2 dataset.

## Project Overview

This project analyzes employee activity to identify potentially suspicious insider behavior using historical user activity and behavioral patterns.

## Technologies Used

- Python
- Apache Spark
- CERT Insider Threat Dataset (r4.2)
- Pandas and Scikit-learn
- Random Forest
- SHAP
- Streamlit

## Project Workflow

1. CERT dataset ingestion and processing.
2. Feature extraction from HTTP, email, file, device, and logon activities.
3. Historical user and peer-group behavioral baselines.
4. Ground-truth labeling using CERT incident records.
5. Random Forest model training.
6. Model evaluation.
7. SHAP-based explainability.
8. Interactive Streamlit dashboard.

## Member Contributions

### Member A – Big Data Processing
- Data ingestion and Spark processing.
- Activity aggregation and feature engineering.
- Historical user and peer-group baselines.
- Feature integration and validation.

### Member B – Machine Learning and Explainability
- Incident parsing and event verification.
- Ground-truth user-day label generation.
- Random Forest model development.
- Model evaluation and confusion matrix.
- SHAP feature-importance analysis.
- Streamlit dashboard development.

## Preliminary Model Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9990 |
| Precision | 0.7460 |
| Recall | 0.9156 |
| F1-score | 0.8222 |
| ROC-AUC | 0.9998 |
| PR-AUC | 0.9392 |

These are preliminary results from a chronological train-test split.

## Explainability

SHAP was used to identify influential behavioral features, including historical HTTP activity, USB activity, file activity, and deviations from peer-group behavior.

## Streamlit Dashboard

The dashboard provides:
- Project Overview
- Model Performance
- Feature Importance
- User-Day Risk Explorer

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py

