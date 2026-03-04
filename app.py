import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Stroke Dashboard", layout="wide")

# Title
st.title("🧠 Stroke Dataset Dashboard")

# Load data
df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Filters")

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
# BASIC CHARTS
# =========================

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Stroke Distribution")
    st.bar_chart(filtered_df["stroke"].value_counts())

with col2:
    st.write("Gender Distribution")
    st.bar_chart(filtered_df["gender"].value_counts())

# =========================
# HISTOGRAMS
# =========================
st.subheader("Health Metrics Distribution")

col1, col2, col3 = st.columns(3)

# AGE HISTOGRAM
with col1:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["age"], bins=20)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution")
    st.pyplot(fig)

# BMI HISTOGRAM
with col2:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["bmi"].dropna(), bins=20)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")
    ax.set_title("BMI Distribution")
    st.pyplot(fig)

# GLUCOSE HISTOGRAM
with col3:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["avg_glucose_level"], bins=20)
    ax.set_xlabel("Glucose Level")
    ax.set_ylabel("Frequency")
    ax.set_title("Glucose Distribution")
    st.pyplot(fig)
# =========================
# HEALTH ANALYSIS
# =========================

st.subheader("Health Factors")

col1, col2 = st.columns(2)

with col1:
    st.write("Hypertension vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["hypertension"], filtered_df["stroke"]))

with col2:
    st.write("Heart Disease vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["heart_disease"], filtered_df["stroke"]))

# =========================
# SMOKING STATUS
# =========================

st.subheader("Smoking Status")
st.bar_chart(filtered_df["smoking_status"].value_counts())

# =========================
# CORRELATION MATRIX
# =========================

st.subheader("Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()

st.dataframe(corr)

# =========================
# DATA PREVIEW
# =========================

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20))
