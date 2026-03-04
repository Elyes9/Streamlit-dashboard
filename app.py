import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="Stroke Dashboard", layout="wide")

st.title("🧠 Stroke Dataset Dashboard")
st.write("Interactive dashboard for exploring the stroke dataset")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------

st.sidebar.header("Filters")

gender = st.sidebar.selectbox("Gender", ["All"] + list(df["gender"].unique()))
smoking = st.sidebar.selectbox("Smoking Status", ["All"] + list(df["smoking_status"].unique()))
work = st.sidebar.selectbox("Work Type", ["All"] + list(df["work_type"].unique()))

age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (20, 80)
)

bmi_range = st.sidebar.slider(
    "BMI Range",
    int(df["bmi"].min()),
    int(df["bmi"].max()),
    (15, 40)
)

filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

if smoking != "All":
    filtered_df = filtered_df[filtered_df["smoking_status"] == smoking]

if work != "All":
    filtered_df = filtered_df[filtered_df["work_type"] == work]

filtered_df = filtered_df[
    (filtered_df["age"].between(age_range[0], age_range[1])) &
    (filtered_df["bmi"].between(bmi_range[0], bmi_range[1]))
]

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------

st.subheader("Key Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))
col2.metric("Average Age", round(filtered_df["age"].mean(),1))
col3.metric("Average BMI", round(filtered_df["bmi"].mean(),1))
col4.metric("Stroke Cases", int(filtered_df["stroke"].sum()))

# -------------------------------------------------
# CATEGORY DISTRIBUTIONS
# -------------------------------------------------

st.subheader("Population Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Gender Distribution")
    st.bar_chart(filtered_df["gender"].value_counts())

with col2:
    st.write("Smoking Status")
    st.bar_chart(filtered_df["smoking_status"].value_counts())

with col3:
    st.write("Work Type")
    st.bar_chart(filtered_df["work_type"].value_counts())

# -------------------------------------------------
# HISTOGRAM FUNCTION (CORRECT X AND Y)
# -------------------------------------------------

def histogram_chart(data, column, bins=20):

    hist, bin_edges = np.histogram(data.dropna(), bins=bins)

    histogram_df = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "count": hist
    })

    histogram_df = histogram_df.set_index("bin_start")

    st.bar_chart(histogram_df)

# -------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------

st.subheader("Health Metrics Distribution")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Age Distribution")
    histogram_chart(filtered_df["age"], "age")

with col2:
    st.write("BMI Distribution")
    histogram_chart(filtered_df["bmi"], "bmi")

with col3:
    st.write("Glucose Level Distribution")
    histogram_chart(filtered_df["avg_glucose_level"], "avg_glucose_level")

# -------------------------------------------------
# HEALTH FACTORS
# -------------------------------------------------

st.subheader("Health Risk Factors")

col1, col2 = st.columns(2)

with col1:
    st.write("Hypertension vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["hypertension"], filtered_df["stroke"]))

with col2:
    st.write("Heart Disease vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["heart_disease"], filtered_df["stroke"]))

# -------------------------------------------------
# AGE VS GLUCOSE
# -------------------------------------------------

st.subheader("Average Glucose by Age")

age_glucose = filtered_df.groupby("age")["avg_glucose_level"].mean()

st.line_chart(age_glucose)

# -------------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------------

st.subheader("Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()

st.dataframe(corr)

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(20))
