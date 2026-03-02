import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Stroke Dataset Dashboard")

# Load dataset
df = pd.read_csv("Uncleaned_Data_Set.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Stroke distribution
st.subheader("Stroke Distribution")
st.bar_chart(df["stroke"].value_counts())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Correlation matrix using matplotlib (without seaborn)
st.subheader("Correlation Matrix")

numeric_df = df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
plt.colorbar(cax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

st.pyplot(fig)
