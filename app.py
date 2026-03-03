import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Stroke Data Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Stroke Dataset Analysis Dashboard")
st.markdown("Interactive exploration of stroke dataset")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

df = df[df["gender"].isin(gender)]

# -------------------------
# METRICS
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(df))
col2.metric("Average Age", round(df["age"].mean(),1))
col3.metric("Average BMI", round(df["bmi"].mean(),1))
col4.metric("Stroke Cases", int(df["stroke"].sum()))

st.divider()

# -------------------------
# AGE DISTRIBUTION
# -------------------------
st.subheader("Age Distribution")

fig = px.histogram(
    df,
    x="age",
    nbins=30,
    title="Age Distribution of Patients"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# BMI DISTRIBUTION
# -------------------------
st.subheader("BMI Distribution")

fig = px.histogram(
    df,
    x="bmi",
    nbins=30,
    title="BMI Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# GENDER DISTRIBUTION
# -------------------------
st.subheader("Gender Distribution")

fig = px.pie(
    df,
    names="gender",
    title="Gender Proportion"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# HYPERTENSION VS STROKE
# -------------------------
st.subheader("Hypertension vs Stroke")

fig = px.bar(
    df,
    x="hypertension",
    color="stroke",
    title="Hypertension and Stroke Relationship"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# HEART DISEASE VS STROKE
# -------------------------
st.subheader("Heart Disease vs Stroke")

fig = px.bar(
    df,
    x="heart_disease",
    color="stroke",
    title="Heart Disease and Stroke"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# WORK TYPE DISTRIBUTION
# -------------------------
st.subheader("Work Type Distribution")

fig = px.histogram(
    df,
    x="work_type",
    color="stroke",
    title="Work Type vs Stroke"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SMOKING STATUS
# -------------------------
st.subheader("Smoking Status")

fig = px.histogram(
    df,
    x="smoking_status",
    color="stroke",
    title="Smoking Status vs Stroke"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# GLUCOSE VS AGE
# -------------------------
st.subheader("Glucose Level vs Age")

fig = px.scatter(
    df,
    x="age",
    y="avg_glucose_level",
    color="stroke",
    title="Age vs Glucose Level"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# BMI VS AGE
# -------------------------
st.subheader("BMI vs Age")

fig = px.scatter(
    df,
    x="age",
    y="bmi",
    color="stroke",
    title="Age vs BMI"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# CORRELATION MATRIX
# -------------------------
st.subheader("Correlation Matrix")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    title="Feature Correlation Matrix"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# DATA PREVIEW
# -------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head(20))
