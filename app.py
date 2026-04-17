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
# LOAD DATA (CACHED)
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_DataSet_Stroke.csv")

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filters")

def safe_unique(col):
    return sorted([x for x in df[col].dropna().unique()])

gender = st.sidebar.selectbox("Gender", ["All"] + safe_unique("gender"))
smoking = st.sidebar.selectbox("Smoking", ["All"] + safe_unique("smoking_status"))
work = st.sidebar.selectbox("Work Type", ["All"] + safe_unique("work_type"))

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
    filtered_df = filtered_df[filtered_df.gender == gender]

if smoking != "All":
    filtered_df = filtered_df[filtered_df.smoking_status == smoking]

if work != "All":
    filtered_df = filtered_df[filtered_df.work_type == work]

filtered_df = filtered_df[
    filtered_df.age.between(*age_range) &
    filtered_df.bmi.between(*bmi_range)
]

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered_df)
stroke_cases = filtered_df.stroke.sum()

col1.metric("Patients", total)
col2.metric("Avg Age", round(filtered_df.age.mean(), 1))
col3.metric("Avg BMI", round(filtered_df.bmi.mean(), 1))
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
    df_plot = pd.DataFrame({"Percent": percent})
    st.bar_chart(df_plot)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gender")
    percentage_bar(filtered_df.gender)

with col2:
    st.subheader("Smoking")
    percentage_bar(filtered_df.smoking_status)

with col3:
    st.subheader("Work Type")
    percentage_bar(filtered_df.work_type)

st.markdown("---")

# -------------------------------------------------
# HISTOGRAM FUNCTION
# -------------------------------------------------
def histogram(data, bins=20):
    hist, bins = np.histogram(data.dropna(), bins=bins)

    df_hist = pd.DataFrame({
        "Value": bins[:-1],
        "Frequency": hist
    }).set_index("Value")

    st.bar_chart(df_hist)

# -------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------
st.markdown("## 🏥 Health Distributions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Age")
    histogram(filtered_df.age)

with col2:
    st.subheader("BMI")
    histogram(filtered_df.bmi)

with col3:
    st.subheader("Glucose")
    histogram(filtered_df.avg_glucose_level)

st.markdown("---")

# -------------------------------------------------
# HEALTH FACTORS
# -------------------------------------------------
st.markdown("## ❤️ Risk Factors vs Stroke")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hypertension")
    st.bar_chart(pd.crosstab(filtered_df.hypertension, filtered_df.stroke))

with col2:
    st.subheader("Heart Disease")
    st.bar_chart(pd.crosstab(filtered_df.heart_disease, filtered_df.stroke))

st.markdown("---")

# -------------------------------------------------
# AGE VS GLUCOSE
# -------------------------------------------------
st.markdown("## 📈 Glucose Trend by Age")

age_glucose = filtered_df.groupby("age")["avg_glucose_level"].mean()
st.line_chart(age_glucose)

st.markdown("---")

# -------------------------------------------------
# CORRELATION MATRIX (FIXED ✅)
# -------------------------------------------------
st.markdown("## 🔗 Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

st.dataframe(corr, use_container_width=True)

st.write("💡 Interpretation:")
st.write("• Values close to 1 → strong positive correlation")
st.write("• Values close to -1 → strong negative correlation")
st.write("• Values near 0 → weak or no relationship")

st.markdown("---")

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
st.markdown("## 🔎 Data Preview")

st.dataframe(filtered_df.head(20), use_container_width=True)
