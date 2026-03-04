import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="Stroke Dashboard", layout="wide")

# -------------------------------------------------
# CUSTOM STYLE
# -------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
}

h1 {
    text-align:center;
    color:#00E5FF;
    font-size:48px;
}

h2, h3 {
    color:#FFD54F;
}

[data-testid="metric-container"] {
    background: linear-gradient(145deg,#1f4037,#99f2c8);
    border-radius:12px;
    padding:18px;
    color:black;
    font-weight:bold;
}

.css-1d391kg {
    background-color:#111827;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.markdown("""
<h1>🧠 Stroke Dataset Analysis Dashboard</h1>
<p style='text-align:center; font-size:18px; color:white;'>
Interactive dashboard for exploring stroke dataset characteristics
</p>
""", unsafe_allow_html=True)

st.markdown("---")

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

st.markdown("## 📊 Key Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))
col2.metric("Average Age", round(filtered_df["age"].mean(),1))
col3.metric("Average BMI", round(filtered_df["bmi"].mean(),1))
col4.metric("Stroke Cases", int(filtered_df["stroke"].sum()))

st.markdown("---")

# -------------------------------------------------
# CATEGORY DISTRIBUTIONS
# -------------------------------------------------

st.markdown("## 👥 Population Overview")

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

st.markdown("---")

# -------------------------------------------------
# HISTOGRAM FUNCTION
# -------------------------------------------------

def histogram_chart(data, bins=20):

    hist, bin_edges = np.histogram(data.dropna(), bins=bins)

    histogram_df = pd.DataFrame({
        "Value": bin_edges[:-1],
        "Frequency": hist
    })

    histogram_df = histogram_df.set_index("Value")

    st.bar_chart(histogram_df)

# -------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------

st.markdown("## 📈 Health Metrics Distribution")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Age Distribution")
    histogram_chart(filtered_df["age"])

with col2:
    st.write("BMI Distribution")
    histogram_chart(filtered_df["bmi"])

with col3:
    st.write("Glucose Level Distribution")
    histogram_chart(filtered_df["avg_glucose_level"])

st.markdown("---")

# -------------------------------------------------
# HEALTH FACTORS
# -------------------------------------------------

st.markdown("## ❤️ Health Risk Factors")

col1, col2 = st.columns(2)

with col1:
    st.write("Hypertension vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["hypertension"], filtered_df["stroke"]))

with col2:
    st.write("Heart Disease vs Stroke")
    st.bar_chart(pd.crosstab(filtered_df["heart_disease"], filtered_df["stroke"]))

st.markdown("---")

# -------------------------------------------------
# AGE VS GLUCOSE
# -------------------------------------------------

st.markdown("## 📉 Average Glucose by Age")

age_glucose = filtered_df.groupby("age")["avg_glucose_level"].mean()

st.line_chart(age_glucose)

st.markdown("---")

# -------------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------------

st.markdown("## 🔗 Correlation Matrix")

numeric_df = filtered_df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()

st.dataframe(corr)

st.markdown("---")

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------

st.markdown("## 🔍 Dataset Preview")

st.dataframe(filtered_df.head(20))
