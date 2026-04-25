import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

saved    = load_model()
model    = saved["model"]
scaler   = saved["scaler"]
features = saved["features"]
metrics  = saved["metrics"]

# ── Header ─────────────────────────────────────────────────
st.title(" Student Exam Score Predictor")
st.markdown("**Multiple Linear Regression** · 6,606 students · 16 features · scikit-learn")
st.divider()

# ── Model Metrics Row ──────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score",       f"{metrics['r2']:.4f}")
col2.metric("RMSE",           f"{metrics['rmse']:.4f}")
col3.metric("MAE",            f"{metrics['mae']:.4f}")
col4.metric("5-Fold CV R²",   f"{metrics['cv_r2']:.4f}")
st.divider()

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [" Predictor", " Feature Importance", " EDA Plots", " About"]
)

# ════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════
with tab1:
    st.subheader(" Predict Exam Score")
    st.markdown("Adjust the inputs below and click **Predict**.")

    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            hours_studied  = st.slider("Hours Studied / week",   1,  44, 20)
            attendance     = st.slider("Attendance (%)",         60, 100, 80)
            previous       = st.slider("Previous Score",         40, 100, 75)
            tutoring       = st.slider("Tutoring Sessions/month", 0,   8,  1)

        with c2:
            physical       = st.slider("Physical Activity (hrs/wk)", 0, 6, 3)
            motivation     = st.selectbox("Motivation Level",
                                          ["Low", "Medium", "High"], index=1)
            parental_inv   = st.selectbox("Parental Involvement",
                                          ["Low", "Medium", "High"], index=1)
            access         = st.selectbox("Access to Resources",
                                          ["Low", "Medium", "High"], index=1)

        with c3:
            income         = st.selectbox("Family Income",
                                          ["Low", "Medium", "High"], index=1)
            teacher        = st.selectbox("Teacher Quality",
                                          ["Low", "Medium", "High"], index=1)
            peer           = st.selectbox("Peer Influence",
                                          ["Negative", "Neutral", "Positive"], index=1)
            parental_edu   = st.selectbox("Parental Education",
                                          ["High School", "College", "Postgraduate"], index=1)

        with c4:
            distance       = st.selectbox("Distance from School",
                                          ["Near", "Moderate", "Far"], index=0)
            internet       = st.selectbox("Internet Access",   ["Yes", "No"], index=0)
            extracurr      = st.selectbox("Extracurricular",   ["Yes", "No"], index=0)
            disabilities   = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # ── Encode inputs ──────────────────────────────────
        ordinal_maps = {
            "Parental_Involvement":     {"Low":0,"Medium":1,"High":2},
            "Access_to_Resources":      {"Low":0,"Medium":1,"High":2},
            "Motivation_Level":         {"Low":0,"Medium":1,"High":2},
            "Family_Income":            {"Low":0,"Medium":1,"High":2},
            "Teacher_Quality":          {"Low":0,"Medium":1,"High":2},
            "Parental_Education_Level": {"High School":0,"College":1,"Postgraduate":2},
            "Distance_from_Home":       {"Near":2,"Moderate":1,"Far":0},
            "Peer_Influence":           {"Negative":0,"Neutral":1,"Positive":2},
        }

        input_raw = {
            "Hours_Studied":              hours_studied,
            "Attendance":                 attendance,
            "Parental_Involvement":       ordinal_maps["Parental_Involvement"][parental_inv],
            "Access_to_Resources":        ordinal_maps["Access_to_Resources"][access],
            "Extracurricular_Activities": 1 if extracurr == "Yes" else 0,
            "Previous_Scores":            previous,
            "Motivation_Level":           ordinal_maps["Motivation_Level"][motivation],
            "Internet_Access":            1 if internet == "Yes" else 0,
            "Tutoring_Sessions":          tutoring,
            "Family_Income":              ordinal_maps["Family_Income"][income],
            "Teacher_Quality":            ordinal_maps["Teacher_Quality"][teacher],
            "Peer_Influence":             ordinal_maps["Peer_Influence"][peer],
            "Physical_Activity":          physical,
            "Learning_Disabilities":      1 if disabilities == "Yes" else 0,
            "Parental_Education_Level":   ordinal_maps["Parental_Education_Level"][parental_edu],
            "Distance_from_Home":         ordinal_maps["Distance_from_Home"][distance],
        }

        # Keep only model features
        input_values = np.array([[input_raw[f] for f in features]])
        input_scaled = scaler.transform(input_values)
        prediction   = model.predict(input_scaled)[0]
        prediction   = np.clip(prediction, 55, 100)

        # ── Display Result ─────────────────────────────────
        st.divider()
        r1, r2, r3 = st.columns([1, 2, 1])

        with r2:
            if prediction >= 85:
                grade, color = " Excellent",        "#2ecc71"
            elif prediction >= 75:
                grade, color = " Very Good",         "#3498db"
            elif prediction >= 65:
                grade, color = " Average",           "#f39c12"
            else:
                grade, color = " Needs Improvement", "#e74c3c"

            st.markdown(
                f"""
                <div style='text-align:center; padding:30px;
                            background:rgba(255,255,255,0.05);
                            border-radius:16px; border:1px solid {color}44'>
                    <div style='font-size:4rem; font-weight:800; color:{color}'>
                        {prediction:.1f}
                    </div>
                    <div style='font-size:1.2rem; color:#aaa; margin-top:6px'>
                        {grade}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int((prediction - 55) / 45 * 100))

        # ── Smart Tips ────────────────────────────────────
        st.subheader(" Improvement Tips")
        tips = []
        if attendance < 80:
            tips.append(f" **Attendance is only {attendance}%** — biggest lever. Each 1% ≈ +0.23 score points.")
        if hours_studied < 20:
            tips.append(f" **Study hours ({hours_studied}/wk) below average (20)** — add 5 more hrs → ~+0.88 points.")
        if motivation == "Low":
            tips.append(" **Low motivation detected** — high motivation students score ~0.75 higher on average.")
        if tutoring < 2:
            tips.append(f" **Add tutoring sessions** — each session/month adds ~+0.59 to predicted score.")
        if access == "Low":
            tips.append(" **Improve resource access** — moving Low → High adds ~+1.46 points.")
        if not tips:
            tips.append(" Your inputs are well-optimized. Keep it up!")

        for tip in tips:
            st.info(tip)

# ════════════════════════════════════════════════
# TAB 2 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════
with tab2:
    st.subheader(" Feature Importance — Standardized Coefficients")

    coef_df = pd.read_csv("coefficients.csv")
    coef_df["Color"] = coef_df["Coefficient"].apply(
        lambda x: "#2ecc71" if x > 0 else "#e74c3c"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        coef_df["Feature"][::-1],
        coef_df["Coefficient"][::-1],
        color=coef_df["Color"][::-1].values
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Feature Importance (after StandardScaler)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **How to read this:**
    - Coefficients are on **standardized** inputs — comparable across features
    - **Green bars** = positive effect on exam score
    - **Red bar** = negative effect (Learning Disabilities)
    - Attendance and Hours Studied dominate all other features
    """)

    st.dataframe(
        coef_df[["Feature", "Coefficient"]].style.format({"Coefficient": "{:.4f}"}),
        use_container_width=True
    )

# ════════════════════════════════════════════════
# TAB 3 — EDA PLOTS
# ════════════════════════════════════════════════
with tab3:
    st.subheader(" Exploratory Data Analysis")

    plot_files = {
        "01 — Target Distribution":    "plots/01_target_distribution.png",
        "02 — Correlation Heatmap":    "plots/02_correlation_heatmap.png",
        "03 — Scatter Top Features":   "plots/03_scatter_top_features.png",
        "04 — Categorical Boxplots":   "plots/04_categorical_boxplots.png",
        "05 — Group Mean Scores":      "plots/05_group_means.png",
        "06 — Actual vs Predicted":    "plots/06_actual_vs_predicted.png",
        "07 — Residual Diagnostics":   "plots/07_residual_diagnostics.png",
        "08 — Feature Importance":     "plots/08_feature_importance.png",
    }

    selected = st.selectbox("Select a plot:", list(plot_files.keys()))
    path = plot_files[selected]

    if os.path.exists(path):
        st.image(path, use_column_width=True)
    else:
        st.warning(f"Plot not found: {path}. Run scripts/04_diagnostics.py first.")

# ════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════
with tab4:
    st.subheader("About This Project")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ###  Project Overview
        Statistical and Predictive Modelling project exploring factors
        that influence student exam performance.

        **Dataset:** 6,607 students, 20 features  
        **Target:** Exam Score (continuous, 55–100)  
        **Model:** Multiple Linear Regression (OLS)

        ###  Methodology
        1. Data Cleaning — mode imputation, outlier removal
        2. EDA — correlations, boxplots, scatterplots
        3. Encoding — ordinal + binary encoding
        4. Feature Selection — Pearson |r| ≥ 0.02
        5. Modelling — Linear Regression (scikit-learn)
        6. Validation — 5-Fold Cross Validation
        7. Diagnostics — residuals, Q-Q, ANOVA, Shapiro-Wilk
        """)

    with c2:
        st.markdown("""
        ###  Key Findings
        | Feature | Coefficient | Rank |
        |---|---|---|
        | Attendance | +2.29 | #1 |
        | Hours Studied | +1.75 | #2 |
        | Access to Resources | +0.73 | #3 |
        | Parental Involvement | +0.71 | #4 |
        | Previous Scores | +0.71 | #5 |
        | Learning Disabilities | -0.27 | Only negative |

        ###  Limitations
        - Shapiro-Wilk rejects normality of residuals
        - CV R² (73.6%) < test R² (82.6%) — mild overfitting
        - Mode imputation may introduce slight bias
        - Linear model assumes linear relationships
        """)

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#aaa; font-size:0.85rem'>
    Built with Python · scikit-learn · Matplotlib · Seaborn · Streamlit
    </div>
    """, unsafe_allow_html=True)