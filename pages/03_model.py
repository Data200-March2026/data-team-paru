import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

st.title("Model Training")
st.markdown("Linear Regression model on student performance data.")

df = pd.read_csv("data/data_cleaned.csv")
st.write(f"Loaded: {df.shape}")

df_model = df.copy()

ordinal_maps = {
    "Parental_Involvement":     {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources":      {"Low": 0, "Medium": 1, "High": 2},
    "Motivation_Level":         {"Low": 0, "Medium": 1, "High": 2},
    "Family_Income":            {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality":          {"Low": 0, "Medium": 1, "High": 2},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home":       {"Near": 2, "Moderate": 1, "Far": 0},
    "Peer_Influence":           {"Negative": 0, "Neutral": 1, "Positive": 2},
}
for col, mapping in ordinal_maps.items():
    df_model[col] = df_model[col].map(mapping)

binary_maps = {
    "Extracurricular_Activities": {"No": 0, "Yes": 1},
    "Internet_Access":            {"No": 0, "Yes": 1},
    "Learning_Disabilities":      {"No": 0, "Yes": 1},
    "Gender":                     {"Male": 0, "Female": 1},
    "School_Type":                {"Public": 0, "Private": 1},
}
for col, mapping in binary_maps.items():
    df_model[col] = df_model[col].map(mapping)

st.success("Encoding complete")

X_all = df_model.drop("Exam_Score", axis=1)
y = df_model["Exam_Score"]

corr_with_target = df_model.corr()["Exam_Score"].drop("Exam_Score").abs()
drop_features = corr_with_target[corr_with_target < 0.02].index.tolist()
X = X_all.drop(columns=drop_features)

st.subheader("Feature Correlations with Exam Score")
st.dataframe(corr_with_target.sort_values(ascending=False).round(4))
st.write(f"Dropped features (|r| < 0.02): {drop_features}")
st.write(f"Features kept: {len(X.columns)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2    = r2_score(y_test, y_pred)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
cv    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")

st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²",           f"{r2:.4f}")
col2.metric("RMSE",         f"{rmse:.4f}")
col3.metric("MAE",          f"{mae:.4f}")
col4.metric("5-Fold CV R²", f"{cv_r2.mean():.4f}")

coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

st.subheader("Feature Coefficients")
st.dataframe(coef_df)

os.makedirs("models", exist_ok=True)
model_data = {
    "model":         model,
    "scaler":        scaler,
    "features":      list(X.columns),
    "metrics": {"r2": r2, "rmse": rmse, "mae": mae,
                "cv_r2": cv_r2.mean(), "cv_std": cv_r2.std()},
    "ordinal_maps":  ordinal_maps,
    "binary_maps":   binary_maps,
    "drop_features": drop_features
}
with open("models/model.pkl", "wb") as f:
    pickle.dump(model_data, f)
coef_df.to_csv("models/coefficients.csv", index=False)
np.save("models/y_test.npy", y_test.values)
np.save("models/y_pred.npy", y_pred)
st.success("Model saved to models/model.pkl")