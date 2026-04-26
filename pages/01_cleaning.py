import pandas as pd
import numpy as np

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv("data.csv")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicates   : {df.duplicated().sum()}")
print(f"\nExam_Score > 100: {(df['Exam_Score'] > 100).sum()} row(s)")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── 2. Remove impossible scoree ─────────────────────────────
df = df[df["Exam_Score"] <= 100].copy()
print(f"\n Removed rows where Exam_Score > 100")
print(f"   New shape: {df.shape}")

# ── 3. Fill missing categorical values with mode ───────────
missing_cols = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]

for col in missing_cols:
    mode_val = df[col].mode()[0]
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(mode_val)
    print(f" Filled '{col}': {missing_count} NaNs → mode = '{mode_val}'")

# ── 4. Strip whitespace from all string columns ────────────
str_cols = df.select_dtypes(include="object").columns
for col in str_cols:
    df[col] = df[col].str.strip()
print(f" Stripped whitespace from {len(str_cols)} string columns")

# ── 5. Final verification ──────────────────────────────────
print("\n" + "=" * 50)
print("CLEANED DATASET")
print("=" * 50)
print(f"Shape          : {df.shape}")
print(f"Missing Values : {df.isnull().sum().sum()}")
print(f"Duplicates     : {df.duplicated().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ── 6. Save cleaned data ───────────────────────────────────
df.to_csv("data_cleaned.csv", index=False)
print("\n Cleaned data saved to: data_cleaned.csv")