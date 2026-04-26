import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ── Setup ──────────────────────────────────────────────────
df = pd.read_csv("data_cleaned.csv")
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

print("Dataset loaded:", df.shape)

# ── Plot 1: Target Distribution + Q-Q ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["Exam_Score"], bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(df["Exam_Score"].mean(), color="red", linestyle="--",
                label=f"Mean = {df['Exam_Score'].mean():.1f}")
axes[0].set_title("Exam Score Distribution", fontweight="bold")
axes[0].set_xlabel("Exam Score")
axes[0].legend()

stats.probplot(df["Exam_Score"], plot=axes[1])
axes[1].set_title("Q-Q Plot: Normality Check", fontweight="bold")

plt.tight_layout()
plt.savefig("plots/01_target_distribution.png", bbox_inches="tight")
plt.close()
print(" Plot 1 saved: Target Distribution")

# ── Plot 2: Correlation Heatmap ────────────────────────────
num_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
            "Previous_Scores", "Tutoring_Sessions",
            "Physical_Activity", "Exam_Score"]

corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap — Numeric Features", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print(" Plot 2 saved: Correlation Heatmap")

print("\nTop correlations with Exam_Score:")
print(corr["Exam_Score"].sort_values(ascending=False).to_string())

# ── Plot 3: Scatter — Top 2 Features ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, col, color in zip(axes,
                           ["Hours_Studied", "Attendance"],
                           ["steelblue", "coral"]):
    ax.scatter(df[col], df["Exam_Score"], alpha=0.15, s=10, color=color)
    m, b = np.polyfit(df[col], df["Exam_Score"], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, m * x_line + b, "k--", linewidth=2,
            label=f"slope = {m:.2f}")
    ax.set_xlabel(col.replace("_", " "), fontweight="bold")
    ax.set_ylabel("Exam Score")
    ax.set_title(f"{col.replace('_', ' ')} vs Exam Score", fontweight="bold")
    ax.legend()

plt.tight_layout()
plt.savefig("plots/03_scatter_top_features.png", bbox_inches="tight")
plt.close()
print(" Plot 3 saved: Scatter Top Features")

# ── Plot 4: Categorical Boxplots ───────────────────────────
cat_features = ["Motivation_Level", "Family_Income",
                "Parental_Involvement", "Teacher_Quality", "Peer_Influence"]
order_map = {
    "Motivation_Level":     ["Low", "Medium", "High"],
    "Family_Income":        ["Low", "Medium", "High"],
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Teacher_Quality":      ["Low", "Medium", "High"],
    "Peer_Influence":       ["Negative", "Neutral", "Positive"],
}

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, cat_features):
    order = order_map.get(col, sorted(df[col].unique()))
    sns.boxplot(data=df, x=col, y="Exam_Score",
                order=order, ax=ax, palette="Set2")
    ax.set_title(col.replace("_", " "), fontweight="bold", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)

plt.suptitle("Exam Score by Categorical Features", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("plots/04_categorical_boxplots.png", bbox_inches="tight")
plt.close()
print(" Plot 4 saved: Categorical Boxplots")

# ── Plot 5: Group Mean Bar Charts ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["School_Type", "Gender", "Learning_Disabilities"]):
    means = df.groupby(col)["Exam_Score"].mean().sort_values(ascending=False)
    bars = ax.bar(means.index, means.values,
                  color=sns.color_palette("Set2", len(means)))
    ax.set_title(f"Avg Score by {col.replace('_', ' ')}", fontweight="bold")
    ax.set_ylim(60, 72)
    ax.set_ylabel("Mean Exam Score")
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plots/05_group_means.png", bbox_inches="tight")
plt.close()
print(" Plot 5 saved: Group Mean Scores")

# ── Print key EDA findings ─────────────────────────────────
print("\n" + "=" * 50)
print("KEY EDA FINDINGS")
print("=" * 50)
print(f"Exam Score — Mean: {df['Exam_Score'].mean():.2f}, "
      f"Std: {df['Exam_Score'].std():.2f}, "
      f"Min: {df['Exam_Score'].min()}, Max: {df['Exam_Score'].max()}")

for col in ["Motivation_Level", "Family_Income"]:
    print(f"\nMean Exam Score by {col}:")
    print(df.groupby(col)["Exam_Score"].mean().sort_values(ascending=False))