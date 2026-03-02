import streamlit as st
import pandas as pd
import seaborn  
import matplotlib.pyplot as plt

st.title("Stroke Dataset Dashboard")

df = pd.read_csv("Uncleaned_Data_Set.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Stroke Distribution")
st.bar_chart(df["stroke"].value_counts())

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=["int64","float64"])

fig, ax = plt.subplots()
seaborn.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)
