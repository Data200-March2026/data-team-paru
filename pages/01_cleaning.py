import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

st.title("Data Cleaning")
st.markdown("Cleaning the student performance dataset.")

df = pd.read_csv("data/data.csv")

st.subheader("Original Dataset")
st.write(f"Shape: {df.shape}")
st.write(f"Missing Values:")
st.dataframe(df.isnull().sum().reset_index().rename(columns={0:"Missing","index":"Column"}))
st.write(f"Duplicates: {df.duplicated().sum()}")
st.write(f"Exam Score > 100: {(df['Exam_Score'] > 100).sum()} row(s)")
st.dataframe(df.head())

df = df[df["Exam_Score"] <= 100].copy()
st.success(f"Removed rows where Exam Score > 100. New shape: {df.shape}")

missing_cols = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
for col in missing_cols:
    mode_val = df[col].mode()[0]
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(mode_val)
    st.write(f"Filled '{col}': {missing_count} NaNs → mode = '{mode_val}'")

str_cols = df.select_dtypes(include="object").columns
for col in str_cols:
    df[col] = df[col].str.strip()

st.subheader("Cleaned Dataset")
st.write(f"Shape: {df.shape}")
st.write(f"Missing Values: {df.isnull().sum().sum()}")
st.write(f"Duplicates: {df.duplicated().sum()}")
st.dataframe(df.describe())

df.to_csv("data/data_cleaned.csv", index=False)
st.success("Cleaned data saved to: data/data_cleaned.csv")