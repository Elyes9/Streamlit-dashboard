import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Stroke Risk Dashboard", layout="wide")

# Title
st.title("🧠 Stroke Risk Analysis Dashboard")

# Load data
df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

# Sidebar filters
st.sidebar.header("Filter Data")

gender = st.sidebar.selectbox("Gender", ["All"] + list(df["gender"].unique()))
smoking = st.sidebar.selectbox("Smoking Status", ["All"] + list(df["smoking_status"].unique()))
work = st.sidebar.selectbox("Work Type", ["All"] + list(df["work_type"].unique()))

filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

if smoking != "All":
    filtered_df = filtered_df[filtered_df["smoking_status"] == smoking]

if work != "All":
    filtered_df = filtered_df[filtered_df["work_type"] == work]

# =========================
# KPI METRICS
# =========================

st.subheader("Key Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))
col2.metric("Average Age", round(filtered_df["age"].mean(),1))
col3.metric("Average BMI", round(filtered_df["bmi"].mean(),1))
col4.metric("Stroke Cases", int(filtered_df["stroke"].sum()))

# =========================
# CHARTS
# =========================

st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("Stroke Distribution")
    st.bar_chart(filtered_df["stroke"].value_counts())

with col2:
    st.write("Gender Distribution")
    st.bar_chart(filtered_df["gender"].value_counts())

# Age distribution
st.subheader("Age Distribution")
st.line_chart(filtered_df["age"])

# BMI distribution
st.subheader("BMI Distribution")
st.line_chart(filtered_df["bmi"].dropna())

# Glucose levels
st.subheader("Average Glucose Level")
st.line_chart(filtered_df["avg_glucose_level"])

# Hypertension vs Stroke
st.subheader("Hypertension vs Stroke")
st.bar_chart(pd.crosstab(filtered_df["hypertension"], filtered_df["stroke"]))

# Heart disease vs Stroke
st.subheader("Heart Disease vs Stroke")
st.bar_chart(pd.crosstab(filtered_df["heart_disease"], filtered_df["stroke"]))

# Smoking status
st.subheader("Smoking Status Distribution")
st.bar_chart(filtered_df["smoking_status"].value_counts())

# =========================
# CORRELATION MATRIX
# =========================

st.subheader("Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()

st.dataframe(corr)

# =========================
# STROKE RISK ANALYSIS
# =========================

st.subheader("Stroke Risk Analysis")

risk_by_age = filtered_df.groupby(pd.cut(filtered_df["age"], bins=5))["stroke"].mean()

st.write("Stroke Probability by Age Group")
st.bar_chart(risk_by_age)

# =========================
# DATA PREVIEW
# =========================

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(20))
