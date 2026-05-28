# Breast Cancer Wisconsin Diagnostic Analysis 

This project documents a comprehensive machine learning pipeline for the classification of breast masses as malignant or benign, utilizing the *Breast Cancer Wisconsin (Diagnostic)* dataset. The objective is to develop a robust diagnostic tool capable of achieving high predictive precision while maintaining clinical safety standards.

## Project Overview
The research includes an exhaustive Exploratory Data Analysis (EDA) and a systematic comparison of 10 different machine learning architectures. By applying advanced feature engineering, data normalization, and rigorous hyperparameter optimization, we identified models that achieve exceptional diagnostic accuracy, with a specific focus on minimizing dangerous False Negatives in an oncological context.

## Pipeline Performance Summary

| Model ID | Architecture | Accuracy | FN | FP |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | Baseline KNN (Raw Data) | 93.86% | 7 | 0 |
| **Model 2** | KNN (StandardScaler) | 96.49% | 4 | 0 |
| **Model 3** | Mean Features Only | 96.49% | 3 | 1 |
| **Model 4** | SE Features Only | 90.35% | 6 | 5 |
| **Model 5** | Top 4 Mean Features | 93.86% | 5 | 2 |
| **Model 6** | Top 10 Smart Features | 97.37% | 3 | 0 |
| **Model 7** | **KNN (K=19 Tuned)** | **98.25%** | **2** | **0** |
| **Model 8** | Decision Tree | 97.37% | 2 | 1 |
| **Model 9** | **Logistic Regression (Winner)**| **99.12%** | **1** | **0** |
| **Model 10** | Random Forest | 97.37% | 3 | 0 |

## Technical Stack
- **Language:** Python 3.13
- **Data Analysis & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.
- **Machine Learning:**
    - **Preprocessing:** `StandardScaler`.
    - **Modeling:** `KNeighborsClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LinearRegression`.
    - **Metrics:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`, `mean_squared_error`, `r2_score`.

## Repository Structure
- `1. breast_cancer_diagnosis_data.ipynb`: Data ingestion, cleaning, and target encoding.
- `2. breast_cancer_diagnosis_EDA.ipynb`: Exploratory analysis, feature distribution, and correlation heatmaps.
- `3. breast_cancer_diangosis_ML.ipynb`: Full modeling pipeline, hyperparameter tuning, and clinical interpretation.

## Clinical Impact & Ethical Considerations
* **Minimizing False Negatives:** In oncology, a False Negative (missing a malignant tumor) is significantly more critical than a False Positive. Our winning model (Logistic Regression) achieved a record-low of 1 False Negative.
* **Triage Optimization:** By achieving zero False Positives in our top-performing models, we ensure that clinicians can trust the system, reducing unnecessary psychological stress for patients and preventing the overloading of hospital resources.
* **Explainability:** This project highlights that data quality and strategic feature selection are the cornerstones of successful clinical AI, surpassing the efficacy of complex "black-box" models.

## Future Work
- **Data Augmentation:** Scaling our dataset with broader demographic representation and multi-institutional data to ensure global generalization and robustness.
- **Advanced Ensembles:** Integrating Gradient Boosting (XGBoost/LightGBM) to capture complex, non-linear relationships even more efficiently.
- **Clinical Validation & Scaling:** The next logical step is to expand the sample size through a multi-center study or design a pilot clinical trial to validate our model's performance on real-time diagnostic workflows, bridging the gap between computational research and bedside application.