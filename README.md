Hospital Readmission Prediction
Overview

Hospital readmissions within 30 days are a big challenge for healthcare systems. They increase costs and sometimes show gaps in patient care.

In this project, I built a predictive model to identify patients who are likely to be readmitted within 30 days. The goal is to help healthcare providers intervene early and improve patient outcomes.

Objective

The main goals of this project were to:

Predict 30-day hospital readmissions for patients

Identify high-risk patients early

Provide actionable insights that healthcare teams can use

Tools & Technologies

For this project, I used:

Python with Pandas and NumPy for data processing

Scikit-learn for building machine learning models

Matplotlib and Seaborn for visualizations

Streamlit to create an interactive dashboard

Git/GitHub for version control and collaboration

Data

I used a de-identified / simulated healthcare dataset. Here’s what it includes:

age – Patient’s age

gender – Male or Female

length_of_stay – Number of days the patient stayed in the hospital

prior_admissions – How many times the patient has been admitted before

chronic_conditions – Number of chronic health conditions

readmitted_30 – Target variable: 1 if readmitted within 30 days, 0 if not

Note: The dataset is anonymized to protect patient privacy. A sample version is available in the data/ folder for reproducibility.

How I Approached the Problem

Data Cleaning & Preprocessing – I handled missing values, encoded categorical variables, and normalized features.

Feature Engineering – Created meaningful features like prior admissions per year and a comorbidity index.

Model Training & Validation – Trained Logistic Regression and Random Forest models and validated them using cross-validation.

Evaluation – Checked model performance using accuracy, recall, precision, and AUC (Area Under the Curve).

Dashboard – Built a simple Streamlit dashboard to visualize predictions and patient risk scores interactively.

Results

The Random Forest model performed the best.

Performance metrics:

Accuracy: 79%

Recall (high-risk patients): 76%

Precision: 74%

AUC: 0.82

The model successfully identifies patients at high risk for readmission, which could help hospitals take early action and improve patient care.

Open the notebook in notebooks/ to explore the analysis.

Run Python scripts in scripts/ to execute the full pipeline.

Launch the Streamlit dashboard:
streamlit run dashboard/streamlit_app.py

Results Visuals

ROC Curve – results/roc_curve.png

Confusion Matrix – results/confusion_matrix.png

These visuals show how well the model predicts high-risk patients.

Disclaimer

This project uses de-identified or simulated patient data for academic purposes.
No real patient information is exposed.
