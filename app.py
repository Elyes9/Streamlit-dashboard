import streamlit as st
import pandas as pd

st.title("Stroke Dataset Dashboard")

df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Stroke Distribution")
st.bar_chart(df["stroke"].value_counts())

st.subheader("Average Glucose Level by Stroke")
st.bar_chart(df.groupby("stroke")["avg_glucose_level"].mean())

st.subheader("BMI Distribution")
st.line_chart(df["bmi"].dropna())
