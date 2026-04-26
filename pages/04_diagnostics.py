import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

st.title("Model Diagnostics")
st.markdown("Residual analysis and hypothesis testing.")

with open("models/model.pkl", "rb") as f:
    saved = pickle.load(f)

y_test    = np.load("models/y_test.npy")
y_pred    = np.load("models/y_pred.npy")
coef_df   = pd.read_csv("models/coefficients.csv")
df        = pd.read_csv("data/data_cleaned.csv")
residuals = y_test - y_pred

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

st.subheader("Plot 6: Actual vs Predicted")
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.25, s=15, color="steelblue")
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Exam Score",    fontweight="bold")
ax.set_ylabel("Predicted Exam Score", fontweight="bold")
ax.set_title("Actual vs Predicted",   fontweight="bold")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.subheader("Plot 7: Residual Diagnostics")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes[0, 0].scatter(y_pred, residuals, alpha=0.2, s=12, color="steelblue")
axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Residuals vs Fitted", fontweight="bold")
stats.probplot(residuals, plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot of Residuals", fontweight="bold")
xr = np.linspace(residuals.min(), residuals.max(), 200)
axes[1, 0].hist(residuals, bins=40, color="steelblue", edgecolor="white", density=True)
axes[1, 0].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
                "r--", linewidth=2, label="Normal Fit")
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Residual Distribution", fontweight="bold")
axes[1, 0].legend()
axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.2, s=12, color="coral")
axes[1, 1].axhline(np.sqrt(np.abs(residuals)).mean(), color="red", linestyle="--")
axes[1, 1].set_xlabel("Fitted Values")
axes[1, 1].set_ylabel("sqrt|Residuals|")
axes[1, 1].set_title("Scale-Location", fontweight="bold")
plt.suptitle("Regression Diagnostics", fontweight="bold", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.subheader("Plot 8: Feature Importance")
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized Coefficient", fontweight="bold")
ax.set_title("Feature Importance", fontweight="bold")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.subheader("Hypothesis Tests")

sample = np.random.choice(residuals, 1000, replace=False)
sw_stat, sw_p = stats.shapiro(sample)
st.write(f"**Shapiro-Wilk:** W = {sw_stat:.4f}, p = {sw_p:.6f}")
st.write(f"Residuals {'ARE' if sw_p > 0.05 else 'are NOT'} normal at a=0.05")

st.write("**Pearson Correlation Tests:**")
for col in ["Attendance", "Hours_Studied", "Previous_Scores"]:
    r, p = stats.pearsonr(df[col], df["Exam_Score"])
    st.write(f"- {col}: r = {r:.4f}, p = {p:.2e}")

st.write("**One-Way ANOVA:**")
for col, levels in [
    ("Motivation_Level", ["Low", "Medium", "High"]),
    ("Family_Income",    ["Low", "Medium", "High"]),
]:
    groups = [df[df[col] == lvl]["Exam_Score"].values for lvl in levels]
    f, p = stats.f_oneway(*groups)
    st.write(f"- {col}: F = {f:.4f}, p = {p:.2e} → {'Significant' if p < 0.05 else 'Not significant'}")