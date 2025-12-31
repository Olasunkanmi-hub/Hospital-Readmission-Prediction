# 30days_admission_full.py
# Full interactive dashboard: analytics + model + patient insights
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

st.set_page_config(layout="wide", page_title="30-Day Readmission — Full Dashboard")

# -------------------------------
# Load data
# -------------------------------
DATA_PATH = "Heart_Disease_Prediction.csv"
data = pd.read_csv(DATA_PATH)

# Prepare labels
data['Sex_str'] = data['Sex'].map({0: 'Female', 1: 'Male'})

le = LabelEncoder()
data['Heart Disease (num)'] = le.fit_transform(data['Heart Disease'])  

# Simulated readmission target
data['readmitted_30days'] = data['Heart Disease (num)']
data['readmitted_label'] = data['readmitted_30days'].map({0: 'No', 1: 'Yes'})

# Sidebar filters
st.sidebar.title("Filters & Model Controls")
age_min, age_max = int(data['Age'].min()), int(data['Age'].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
sex_choices = st.sidebar.multiselect("Sex", options=['Female','Male'], default=['Female','Male'])

st.sidebar.markdown("---")
st.sidebar.subheader("Model settings")
use_scaling = st.sidebar.checkbox("Scale numeric features", value=True)
rf_estimators = st.sidebar.slider("RandomForest n_estimators", 50, 500, 100, step=50)
train_test_split_ratio = st.sidebar.slider("Test set proportion", 0.1, 0.4, 0.2, step=0.05)

# Filter dataset
filtered = data[
    (data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1]) &
    (data['Sex_str'].isin(sex_choices))
].copy()

st.title("30-Day Hospital Readmission — Full Interactive Dashboard")
st.markdown("This dashboard shows exploratory visuals, model results, and patient-level insights.")

# ---------------------------
# Feature list and modeling
# ---------------------------
feature_cols = [
    'Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120','EKG results',
    'Max HR','Exercise angina','ST depression','Slope of ST','Number of vessels fluro','Thallium'
]

X = data[feature_cols].copy()
y = data['readmitted_30days'].copy()

categorical_cols = ['Chest pain type','FBS over 120','EKG results','Exercise angina',
                    'Slope of ST','Number of vessels fluro','Thallium']

X_model = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=train_test_split_ratio, random_state=42, stratify=y
)

numeric_cols = ['Age','BP','Cholesterol','Max HR','ST depression']
if use_scaling:
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Models
rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=2000, solver='lbfgs')
lr.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

# -----------------------
# Layout
# -----------------------
col1, col2 = st.columns((1,1))

with col1:
    st.header("Exploratory Visuals")

    # Violin plot
    st.subheader("1) Age vs Readmission")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.violinplot(
        x='readmitted_label', y='Age', data=filtered,
        color='mediumseagreen', inner='quartile', ax=ax
    )
    st.pyplot(fig)

    # Sex CountPlot
    st.subheader("2) Readmission by Sex")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.countplot(
        x='Sex_str', hue='readmitted_label', data=filtered,
        palette={'No':'lightgray','Yes':'salmon'}, ax=ax2
    )
    st.pyplot(fig2)

    # Age–Sex Heatmap
    st.subheader("3) Age–Sex Heatmap")
    filtered['Age_bin'] = pd.cut(
        filtered['Age'], bins=[20,30,40,50,60,70,80],
        labels=['20s','30s','40s','50s','60s','70s+'], right=False
    )
    pivot = filtered.pivot_table(
        index='Age_bin', columns='Sex_str',
        values='readmitted_30days', aggfunc='mean'
    )
    fig3 = px.imshow(pivot, text_auto='.2f', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.header("Model Performance")

    # Confusion matrix
    st.subheader("4) Confusion Matrix (Random Forest)")
    cm = confusion_matrix(y_test, y_pred_rf)
    fig4, ax4 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No','Yes'], yticklabels=['No','Yes'], ax=ax4)
    st.pyplot(fig4)

    # ROC Curve
    st.subheader("5) ROC Curve (RF vs LR)")
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    fig5, ax5 = plt.subplots(figsize=(6,4))
    ax5.plot(fpr_rf, tpr_rf, label=f'RF AUC={auc(fpr_rf, tpr_rf):.2f}')
    ax5.plot(fpr_lr, tpr_lr, label=f'LR AUC={auc(fpr_lr, tpr_lr):.2f}')
    ax5.plot([0,1],[0,1],'k--')
    ax5.legend()
    st.pyplot(fig5)

    # RF Feature Importance
    st.subheader("6) Feature Importance (Random Forest)")
    fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
    fig6 = px.bar(fi, orientation='h')
    st.plotly_chart(fig6, use_container_width=True)

# -----------------------
# Additional analytics
# -----------------------
st.markdown("---")
st.header("Additional Analytical Views")

a1, a2 = st.columns((1,1))

with a1:
    # Risk score distribution
    st.subheader("Risk Score Distribution")
    fig_risk = px.histogram(
        pd.DataFrame({'prob': y_prob_rf}), x='prob', nbins=25
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Donut: Readmission proportion
    st.subheader("Readmission Proportion")
    counts = filtered['readmitted_label'].value_counts()
    fig_donut = px.pie(
        names=counts.index, values=counts.values, hole=0.5,
        color=counts.index,
        color_discrete_map={'No':'lightgray','Yes':'salmon'}
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with a2:
    st.subheader("Feature Impact Proxies")
    proxies = ['FBS over 120','Exercise angina','Number of vessels fluro',
               'Chest pain type','Thallium']

    for col in proxies:
        if col in data.columns:
            pivot_p = filtered.groupby(col)['readmitted_30days'].mean().reset_index()
            figp = px.bar(
                pivot_p, x=col, y='readmitted_30days',
                labels={'readmitted_30days':'Readmission rate'},
                title=f'Readmission Rate by {col}'
            )
            st.plotly_chart(figp, use_container_width=True)

# -----------------------
# Patient-level insights
# -----------------------
st.markdown("---")
st.header("Patient-level Insights")

idx = st.number_input(
    "Select patient index", 
    min_value=int(data.index.min()), 
    max_value=int(data.index.max()), 
    value=int(data.index.min())
)

patient = data.loc[idx]

st.subheader("Patient Details")
st.write(patient[feature_cols + ['Heart Disease','readmitted_label','Sex_str']].to_frame().T)

# Prepare patient for model
patient_X = data.loc[[idx], feature_cols].copy()
patient_X_model = pd.get_dummies(patient_X, columns=categorical_cols, drop_first=True)

for c in X_train.columns:
    if c not in patient_X_model.columns:
        patient_X_model[c] = 0

patient_X_model = patient_X_model[X_train.columns]

if use_scaling:
    patient_X_model[numeric_cols] = scaler.transform(patient_X_model[numeric_cols])

prob_patient_rf = rf.predict_proba(patient_X_model)[:,1][0]
prob_patient_lr = lr.predict_proba(patient_X_model)[:,1][0]

st.subheader("Predicted Readmission Probability")
st.metric("Random Forest", f"{prob_patient_rf:.2f}")
st.metric("Logistic Regression", f"{prob_patient_lr:.2f}")

# -----------------------
# Model evaluation text
# -----------------------
st.markdown("---")
st.header("Model Evaluation Summary")

colA, colB = st.columns(2)

with colA:
    st.subheader("Random Forest")
    st.text(classification_report(y_test, y_pred_rf, digits=3))

with colB:
    st.subheader("Logistic Regression")
    st.text(classification_report(y_test, y_pred_lr, digits=3))


st.caption("This demo uses Heart Disease as a proxy target for 30-day readmission.")