import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Stroke Dashboard", layout="wide")

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
.main {background-color: #0E1117;}
h1, h2, h3 {color: #4CAF50;}

[data-testid="metric-container"] {
    background-color: #1c1f26;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #2d3139;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("""
<h1 style='text-align: center;'>🧠 Stroke Dashboard</h1>
<p style='text-align: center;'>Advanced Interactive Analysis</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_DataSet_Stroke.csv")

df = load_data()

# -------------------------------------------------
# 🔥 ROBUST CLEANING (FINAL FIX)
# -------------------------------------------------

# -------- GENDER --------
df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
df["gender"] = df["gender"].map({0: "Female", 1: "Male"})

# -------- HEALTH FLAGS --------
for col in ["hypertension", "heart_disease", "stroke"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].map({0: "No", 1: "Yes"})

# -------- SMOKING --------
df["smoking_status"] = df["smoking_status"].astype(str).str.strip().str.lower()
df["smoking_status"] = df["smoking_status"].map({
    "never smoked": "Never Smoked",
    "formerly smoked": "Former Smoker",
    "smokes": "Smoker",
    "unknown": "Unknown"
})

# -------- WORK TYPE --------
df["work_type"] = df["work_type"].astype(str).str.strip()
df["work_type"] = df["work_type"].map({
    "Private": "Private Sector",
    "Self-employed": "Self Employed",
    "Govt_job": "Government Job",
    "children": "Children",
    "Never_worked": "Never Worked"
})

# Remove rows with invalid categories
df = df.dropna(subset=["gender", "smoking_status", "work_type"])

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filters")

def safe_unique(col):
    return sorted(df[col].dropna().unique())

gender = st.sidebar.selectbox("Gender", ["All"] + list(safe_unique("gender")))
smoking = st.sidebar.selectbox("Smoking", ["All"] + list(safe_unique("smoking_status")))
work = st.sidebar.selectbox("Work Type", ["All"] + list(safe_unique("work_type")))

age_range = st.sidebar.slider(
    "Age",
    int(df.age.min()),
    int(df.age.max()),
    (20, 80)
)

bmi_range = st.sidebar.slider(
    "BMI",
    int(df.bmi.min()),
    int(df.bmi.max()),
    (15, 40)
)

# -------------------------------------------------
# FILTER DATA
# -------------------------------------------------
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

if smoking != "All":
    filtered_df = filtered_df[filtered_df["smoking_status"] == smoking]

if work != "All":
    filtered_df = filtered_df[filtered_df["work_type"] == work]

filtered_df = filtered_df[
    filtered_df["age"].between(*age_range) &
    filtered_df["bmi"].between(*bmi_range)
]

# -------------------------------------------------
# KPI
# -------------------------------------------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered_df)
stroke_cases = (filtered_df["stroke"] == "Yes").sum()

col1.metric("Patients", total)
col2.metric("Avg Age", round(filtered_df["age"].mean(), 1))
col3.metric("Avg BMI", round(filtered_df["bmi"].mean(), 1))
col4.metric("Stroke Cases", int(stroke_cases))
col5.metric("Stroke Rate (%)", round((stroke_cases / total)*100, 2) if total > 0 else 0)

st.markdown("---")

# -------------------------------------------------
# DISTRIBUTIONS
# -------------------------------------------------
st.markdown("## 📊 Population Overview")

def percentage_bar(series):
    counts = series.value_counts()
    percent = (counts / counts.sum()) * 100
    st.bar_chart(percent)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gender")
    percentage_bar(filtered_df["gender"])

with col2:
    st.subheader("Smoking")
    percentage_bar(filtered_df["smoking_status"])

with col3:
    st.subheader("Work Type")
    percentage_bar(filtered_df["work_type"])

st.markdown("---")

# -------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------
def histogram(data, bins=20):
    hist, bins = np.histogram(data.dropna(), bins=bins)
    df_hist = pd.DataFrame({"Value": bins[:-1], "Freq": hist}).set_index("Value")
    st.bar_chart(df_hist)

st.markdown("## 🏥 Health Distributions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Age")
    histogram(filtered_df["age"])

with col2:
    st.subheader("BMI")
    histogram(filtered_df["bmi"])

with col3:
    st.subheader("Glucose")
    histogram(filtered_df["avg_glucose_level"])

st.markdown("---")

# -------------------------------------------------
# RISK FACTORS
# -------------------------------------------------
st.markdown("## ❤️ Risk Factors vs Stroke")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hypertension")
    st.bar_chart(pd.crosstab(filtered_df["hypertension"], filtered_df["stroke"]))

with col2:
    st.subheader("Heart Disease")
    st.bar_chart(pd.crosstab(filtered_df["heart_disease"], filtered_df["stroke"]))

st.markdown("---")

# -------------------------------------------------
# TREND
# -------------------------------------------------
st.markdown("## 📈 Glucose Trend by Age")

trend = filtered_df.groupby("age")["avg_glucose_level"].mean()
st.line_chart(trend)

st.markdown("---")

# -------------------------------------------------
# CORRELATION
# -------------------------------------------------
st.markdown("## 🔗 Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

st.dataframe(corr, use_container_width=True)

st.markdown("---")

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
st.markdown("## 🔎 Data Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)
