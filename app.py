import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stat-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-card h2 { margin: 0; font-size: 26px; color: #f9fafb; }
    .stat-card p { margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; }
    .result-box {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        border: 1px solid #10b981;
    }
    .result-box h1 { color: #ecfdf5; font-size: 46px; margin: 0; }
    .result-box p { color: #a7f3d0; margin-top: 6px; font-size: 15px; }
    .factor-pill {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 999px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 13px;
        color: #d1d5db;
    }
    section[data-testid="stSidebar"] { background-color: #111827; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("Insurance_Company_Cost_Prediction_Model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

FEATURE_ORDER = [
    "age", "bmi", "children", "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
]

# Metrics captured from the project notebook (train/test split, 80/20, random_state=42)
MODEL_METRICS = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression (Deployed)"],
        "R2 Score": [0.8069, 0.8060, 0.8069],
        "MAE": [4177.05, 4194.01, 4177.89],
        "RMSE": [5956.34, 5971.34, 5957.43],
    }
)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("🏥 Medical Insurance Cost Predictor")
st.caption(
    "A regression-based dashboard that estimates annual medical insurance charges "
    "from demographic and lifestyle information — built on a Lasso Regression model."
)

tab_predict, tab_insights, tab_models = st.tabs(
    ["💰 Predict", "📊 Feature Insights", "🧪 Model Comparison"]
)

# --------------------------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------------------------
st.sidebar.header("👤 Applicant Details")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30, step=1)
bmi = st.sidebar.slider("BMI (Body Mass Index)", min_value=15.0, max_value=53.0, value=26.5, step=0.1)
children = st.sidebar.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5], index=0)
sex = st.sidebar.radio("Sex", options=["Female", "Male"], horizontal=True)
smoker = st.sidebar.radio("Smoker", options=["No", "Yes"], horizontal=True)
region = st.sidebar.selectbox(
    "Region", options=["Northeast", "Northwest", "Southeast", "Southwest"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: **Lasso Regression** · Trained on the classic medical insurance "
    "charges dataset (age, sex, BMI, children, smoker, region)."
)

# --------------------------------------------------------------------------------------
# Build input row in the exact order the model expects
# --------------------------------------------------------------------------------------
def build_input_row():
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "Male" else 0,
        "smoker_yes": 1 if smoker == "Yes" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


input_df = build_input_row()
prediction = float(model.predict(input_df)[0])
prediction = max(prediction, 0)

# --------------------------------------------------------------------------------------
# TAB 1 — Predict
# --------------------------------------------------------------------------------------
with tab_predict:
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.subheader("Applicant Snapshot")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-card"><h2>{age}</h2><p>Age (years)</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f'<div class="stat-card"><h2>{children}</h2><p>Children</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><h2>{bmi:.1f}</h2><p>BMI</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f'<div class="stat-card"><h2>{region}</h2><p>Region</p></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown(
            f'<span class="factor-pill">Sex: {sex}</span>'
            f'<span class="factor-pill">Smoker: {smoker}</span>',
            unsafe_allow_html=True,
        )

        bmi_label = (
            "Underweight" if bmi < 18.5 else
            "Normal" if bmi < 25 else
            "Overweight" if bmi < 30 else
            "Obese"
        )
        st.info(f"BMI Category: **{bmi_label}**")

        if smoker == "Yes":
            st.warning(
                "Smoking status is the single strongest driver of insurance cost "
                "in this model — expect a large premium increase."
            )

    with col_right:
        st.subheader("Predicted Annual Insurance Charge")
        st.markdown(
            f"""
            <div class="result-box">
                <p>Estimated Charge</p>
                <h1>${prediction:,.2f}</h1>
                <p>per year, based on Lasso Regression</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        low, high = prediction * 0.85, prediction * 1.15
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"prefix": "$", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, max(60000, prediction * 1.3)]},
                    "bar": {"color": "#10b981"},
                    "steps": [
                        {"range": [0, 15000], "color": "#1f2937"},
                        {"range": [15000, 35000], "color": "#374151"},
                        {"range": [35000, max(60000, prediction * 1.3)], "color": "#4b5563"},
                    ],
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )
        fig.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f9fafb"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Rough plausible range: **${low:,.0f} – ${high:,.0f}** "
            "(illustrative, not a confidence interval)."
        )

# --------------------------------------------------------------------------------------
# TAB 2 — Feature Insights
# --------------------------------------------------------------------------------------
with tab_insights:
    st.subheader("What drives the prediction?")
    st.write(
        "Coefficients from the deployed Lasso Regression model. A positive coefficient "
        "pushes predicted charges up; a negative coefficient pulls them down. Larger "
        "magnitude = stronger influence."
    )

    coef_df = pd.DataFrame(
        {"Feature": FEATURE_ORDER, "Coefficient": model.coef_}
    ).sort_values("Coefficient", key=abs, ascending=True)

    fig_imp = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale=["#ef4444", "#374151", "#10b981"],
    )
    fig_imp.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### This applicant's contribution breakdown")
    contrib = input_df.iloc[0].values * model.coef_
    contrib_df = pd.DataFrame({"Feature": FEATURE_ORDER, "Contribution": contrib})
    contrib_df = contrib_df[contrib_df["Contribution"] != 0].sort_values(
        "Contribution", key=abs, ascending=True
    )
    if not contrib_df.empty:
        fig_contrib = px.bar(
            contrib_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#ef4444", "#374151", "#10b981"],
        )
        fig_contrib.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f9fafb"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
    st.caption(
        f"Base charge (intercept): **${model.intercept_:,.2f}** — added to every prediction "
        "before the feature effects above are applied."
    )

# --------------------------------------------------------------------------------------
# TAB 3 — Model Comparison
# --------------------------------------------------------------------------------------
with tab_models:
    st.subheader("Regression Models Evaluated")
    st.write(
        "Three regression models were trained and compared during the project "
        "(80/20 train-test split). **Lasso Regression** was selected for deployment "
        "for its strong performance and built-in feature selection via L1 regularization."
    )

    m1, m2, m3 = st.columns(3)
    deployed = MODEL_METRICS.iloc[2]
    with m1:
        st.markdown(
            f'<div class="stat-card"><h2>{deployed["R2 Score"]:.3f}</h2><p>R² Score</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="stat-card"><h2>${deployed["MAE"]:,.0f}</h2><p>MAE</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="stat-card"><h2>${deployed["RMSE"]:,.0f}</h2><p>RMSE</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    fig_r2 = px.bar(
        MODEL_METRICS, x="Model", y="R2 Score", color="Model",
        color_discrete_sequence=["#6366f1", "#f59e0b", "#10b981"],
        text_auto=".3f",
    )
    fig_r2.update_layout(
        height=380, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb"}, yaxis_range=[0.7, 0.85],
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(
            MODEL_METRICS, x="Model", y="MAE", color="Model",
            color_discrete_sequence=["#6366f1", "#f59e0b", "#10b981"],
            text_auto=".0f",
        )
        fig_mae.update_layout(
            height=340, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f9fafb"},
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    with c2:
        fig_rmse = px.bar(
            MODEL_METRICS, x="Model", y="RMSE", color="Model",
            color_discrete_sequence=["#6366f1", "#f59e0b", "#10b981"],
            text_auto=".0f",
        )
        fig_rmse.update_layout(
            height=340, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f9fafb"},
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.dataframe(MODEL_METRICS.set_index("Model"), use_container_width=True)

    st.markdown(
        """
        **Why Lasso Regression was chosen:** all three models performed similarly,
        but Lasso achieved the best overall balance of R² Score and prediction error
        while also shrinking less-useful feature coefficients toward zero,
        effectively performing automatic feature selection.
        """
    )

st.markdown("---")
st.caption(
    "Built with Streamlit · Regression model trained on the Medical Insurance Cost "
    "dataset (age, sex, BMI, children, smoker, region) · For educational/project use only."
)
