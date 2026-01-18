Hospital Readmission Prediction

This project uses Python and machine learning to predict whether a patient is likely to be readmitted to the hospital within 30 days of discharge. The goal is to support healthcare decision-making by identifying high-risk patients early.

Tools & Technologies

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

Streamlit

Dataset

The dataset contains anonymized patient demographic and clinical information related to hospital admissions.
Target variable: 30-day hospital readmission (Yes/No).

Approach

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature encoding and scaling

Machine learning model training

Model evaluation using accuracy, precision, recall, and confusion matrix

Visualization and interaction using a Streamlit dashboard

Results

The model is able to classify patients based on their likelihood of readmission and highlights key factors contributing to readmission risk. This demonstrates how predictive analytics can be applied in healthcare to reduce avoidable readmissions.

Streamlit Dashboard

An interactive Streamlit dashboard is included to visualize predictions and model outputs, making the results accessible to non-technical users.

How to Run
git clone https://github.com/Olasunkanmi-hub/Hospital-Readmission-Prediction.git
cd Hospital-Readmission-Prediction
pip install -r requirements.txt
streamlit run app.py

Author

Olasunkanmi Oladele
GitHub: https://github.com/Olasunkanmi-hub
