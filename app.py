import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Stroke Dashboard", layout="wide")

# Title
st.title("🧠 Stroke Data Analysis Dashboard")

# Load dataset
df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

# Sidebar filters
st.sidebar.header("Filters")

gender = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + list(df["gender"].unique())
)

if gender != "All":
    df = df[df["gender"] == gender]

# KPIs
st.subheader("Key Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(df))
col2.metric("Average Age", round(df["age"].mean(),1))
col3.metric("Stroke Cases", int(df["stroke"].sum()))

# Charts
st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("Stroke Distribution")
    st.bar_chart(df["stroke"].value_counts())

with col2:
    st.write("Age Distribution")
    st.line_chart(df["age"])

# More curves
st.subheader("Health Indicators")

col3, col4 = st.columns(2)

with col3:
    st.write("Average Glucose Level")
    st.line_chart(df["avg_glucose_level"])

with col4:
    st.write("BMI Distribution")
    st.line_chart(df["bmi"].dropna())

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(20))
