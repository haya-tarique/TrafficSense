
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
import shap
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrafficSense",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM DARK THEME
# IMPORTANT: Keep this inside ONE st.markdown("""...""")
# so Streamlit renders the CSS instead of showing it as text.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(255, 83, 100, 0.10), transparent 26%),
            radial-gradient(circle at 15% 80%, rgba(70, 110, 255, 0.06), transparent 30%),
            #070b11;
        color: #f5f7fa;
    }

    .block-container {
        max-width: 1450px;
        padding: 2rem 3rem 4rem 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: #090e15;
        border-right: 1px solid #202833;
    }

    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.4rem;
    }

    .brand {
        font-size: 29px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .brand span {
        color: #ff5364;
    }

    .brand-subtitle {
        color: #718096;
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 28px;
    }

    .nav-label {
        color: #66758a;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 24px 0 10px 0;
    }

    div[data-testid="stRadio"] label {
        color: #b7c0ce !important;
        font-weight: 600;
        border-radius: 10px;
        padding: 8px 10px;
    }

    div[data-testid="stRadio"] label:hover {
        background: #141b25;
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #101722, #0b1018);
        border: 1px solid #202a36;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    div[data-testid="stMetricLabel"] {
        color: #718096 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(18,25,35,0.96), rgba(9,14,21,0.96));
        border: 1px solid #202a36;
        border-radius: 18px;
    }

    .page-title {
        font-size: 40px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1.2px;
        margin-bottom: 6px;
    }

    .page-description {
        color: #718096;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .section-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #718096;
        font-size: 13px;
        margin-bottom: 16px;
    }

    .hero {
        background: linear-gradient(135deg, #101722, #0b1018);
        border: 1px solid #202a36;
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 28px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #718096;
        font-size: 13px;
    }

    .status-online {
        color: #45e0a3;
        font-weight: 700;
        font-size: 13px;
    }

    .feature-card {
        background: #0e151f;
        border: 1px solid #202a36;
        border-radius: 14px;
        padding: 18px;
        min-height: 125px;
    }

    .feature-number {
        color: #ff5364;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .feature-name {
        color: #aeb8c7;
        font-size: 12px;
        min-height: 34px;
    }

    .feature-value {
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
        margin-top: 8px;
    }

    .forecast-card {
        background: linear-gradient(145deg, #111925, #0b1018);
        border: 1px solid #242f3d;
        border-radius: 20px;
        padding: 28px;
    }

    .small-label {
        color: #718096;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .big-number {
        color: #ffffff;
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -2px;
        margin-top: 8px;
    }

    .info-card {
        background: #0e151f;
        border: 1px solid #202a36;
        border-radius: 16px;
        padding: 22px;
    }

    .info-label {
        color: #718096;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .info-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
    }

    .footer-custom {
        text-align: center;
        color: #4f5d70;
        font-size: 12px;
        padding: 45px 0 10px;
    }

    .stButton > button {
        width: 100%;
        background: #111925;
        color: #d9e0ea;
        border: 1px solid #263241;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #ff5364;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    feature_file = Path("data/processed_features.npz")

    if not feature_file.exists():
        st.error(
            "processed_features.npz was not found. "
            "Run prepare_features.py first."
        )
        st.stop()

    with np.load(feature_file) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    return X_train, y_train, X_test, y_test


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = Path("models/xgboost_traffic_model.pkl")

    if not model_path.exists():
        st.error(
            "Model not found at models/xgboost_traffic_model.pkl"
        )
        st.stop()

    return joblib.load(model_path)


# ============================================================
# CONGESTION CLASSIFICATION
# ============================================================

def classify_traffic(value):
    if value < 0.1971:
        return "LOW"
    elif value < 0.4040:
        return "MEDIUM"
    return "HIGH"


# ============================================================
# INITIALIZE
# ============================================================

X_train, y_train, X_test, y_test = load_data()
model = load_model()

predictions = np.asarray(model.predict(X_test))

feature_names = [
    "Last Traffic Value",
    "Average Traffic",
    "Traffic Variation",
    "Minimum Traffic",
    "Maximum Traffic",
]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">Traffic<span>Sense</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">AI-powered traffic intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Traffic Forecast",
            "Analytics",
            "Model Insights",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">SYSTEM STATUS</div>',
        unsafe_allow_html=True,
    )

    st.success("● MODEL ONLINE")
    st.caption("XGBoost prediction engine is active")


# ============================================================
# GLOBAL OBSERVATION
# ============================================================

if "global_observation" not in st.session_state:
    st.session_state.global_observation = 1

observation = st.session_state.global_observation
index = observation - 1

selected_features = X_test[index].reshape(1, -1)
prediction = float(model.predict(selected_features)[0])
actual = float(y_test[index])
difference = prediction - actual
classification = classify_traffic(prediction)


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">TrafficSense Intelligence Platform</div>
        <div class="hero-subtitle">
            Traffic forecasting · congestion analytics · model explainability
        </div>
        <div class="status-online" style="margin-top:12px;">
            ● SYSTEM ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="page-title">Traffic intelligence at a glance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Monitor traffic conditions, model performance and congestion trends.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("R² SCORE", "84.37%", "Model performance")

    with c2:
        st.metric("RMSE", "0.0809", "Prediction error")

    with c3:
        st.metric("MAE", "0.0618", "Average error")

    with c4:
        st.metric("TEST OBSERVATIONS", f"{len(X_test):,}")

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:
        with st.container(border=True):
            st.subheader("Current traffic forecast")

            st.caption(f"Observation #{observation:,}")

            st.metric(
                "Predicted intensity",
                f"{prediction:.4f}",
            )

            if classification == "LOW":
                st.success("🟢 LOW CONGESTION")
            elif classification == "MEDIUM":
                st.warning("🟡 MEDIUM CONGESTION")
            else:
                st.error("🔴 HIGH CONGESTION")

            st.progress(min(max(prediction, 0.0), 1.0))

            a, b = st.columns(2)

            with a:
                st.caption("Actual")
                st.write(f"### {actual:.4f}")

            with b:
                st.caption("Prediction difference")
                st.write(f"### {difference:+.4f}")

    with right:
        with st.container(border=True):
            st.subheader("Traffic signals")

            feature_df = pd.DataFrame(
                {
                    "Signal": feature_names,
                    "Value": [
                        round(float(x), 4)
                        for x in selected_features[0][:5]
                    ],
                }
            )

            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True,
            )

    st.write("")

    with st.container(border=True):
        st.subheader("Traffic activity")

        trend_size = min(500, len(y_test))

        chart_df = pd.DataFrame(
            {
                "Observation": np.arange(1, trend_size + 1),
                "Traffic": y_test[:trend_size],
            }
        )

        chart = (
            alt.Chart(chart_df)
            .mark_area(
                line=True,
                opacity=0.25,
            )
            .encode(
                x=alt.X("Observation:Q", title="Observation"),
                y=alt.Y("Traffic:Q", title="Traffic intensity"),
                tooltip=[
                    "Observation",
                    alt.Tooltip("Traffic:Q", format=".4f"),
                ],
            )
            .properties(height=350)
            .interactive()
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )


# ============================================================
# TRAFFIC FORECAST
# ============================================================

elif page == "Traffic Forecast":

    st.markdown(
        '<div class="page-title">Traffic Forecast</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Select an observation and generate an AI-powered traffic forecast.'
        '</div>',
        unsafe_allow_html=True,
    )

    observation = st.slider(
        "Select traffic observation",
        min_value=1,
        max_value=len(X_test),
        value=st.session_state.global_observation,
        key="forecast_slider",
    )

    st.session_state.global_observation = observation

    index = observation - 1
    selected_features = X_test[index].reshape(1, -1)
    prediction = float(model.predict(selected_features)[0])
    actual = float(y_test[index])
    difference = prediction - actual
    classification = classify_traffic(prediction)

    left, right = st.columns([1.35, 1])

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="small-label">PREDICTED TRAFFIC INTENSITY</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="big-number">{prediction:.4f}</div>',
                unsafe_allow_html=True,
            )

            if classification == "LOW":
                st.success("🟢 LOW CONGESTION — Traffic is relatively clear.")
            elif classification == "MEDIUM":
                st.warning("🟡 MEDIUM CONGESTION — Moderate congestion expected.")
            else:
                st.error("🔴 HIGH CONGESTION — Heavy congestion expected.")

            st.progress(min(max(prediction, 0.0), 1.0))

    with right:
        with st.container(border=True):
            st.subheader("Prediction analysis")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("PREDICTED", f"{prediction:.4f}")

            with c2:
                st.metric("ACTUAL", f"{actual:.4f}")

            st.divider()

            st.metric(
                "ABSOLUTE ERROR",
                f"{abs(difference):.4f}",
            )

            if difference > 0:
                st.caption("Prediction is above the actual value.")
            elif difference < 0:
                st.caption("Prediction is below the actual value.")
            else:
                st.caption("Prediction exactly matches the actual value.")

    st.write("")

    st.markdown(
        '<div class="section-title">Model input signals</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Features supplied to the XGBoost prediction model.'
        '</div>',
        unsafe_allow_html=True,
    )

    feature_cols = st.columns(5)

    for i, col in enumerate(feature_cols):
        with col:
            value = float(selected_features[0][i])

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-number">0{i + 1}</div>
                    <div class="feature-name">{feature_names[i]}</div>
                    <div class="feature-value">{value:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    with st.container(border=True):
        st.subheader("AI forecast interpretation")

        st.write(
            f"The model predicts a traffic intensity of "
            f"{prediction:.4f}, which corresponds to "
            f"{classification} congestion."
        )

        st.caption(
            "The prediction is generated from the five engineered "
            "traffic features shown above."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="page-title">Traffic Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Explore congestion distribution, traffic statistics and model predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    classes = [
        classify_traffic(float(x))
        for x in predictions
    ]

    counts = pd.Series(classes).value_counts()

    low = int(counts.get("LOW", 0))
    medium = int(counts.get("MEDIUM", 0))
    high = int(counts.get("HIGH", 0))
    total = low + medium + high

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("TOTAL", f"{total:,}")

    with c2:
        st.metric("LOW", f"{low:,}")

    with c3:
        st.metric("MEDIUM", f"{medium:,}")

    with c4:
        st.metric("HIGH", f"{high:,}")

    st.write("")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Congestion distribution")

            donut_df = pd.DataFrame(
                {
                    "Level": ["LOW", "MEDIUM", "HIGH"],
                    "Count": [low, medium, high],
                }
            )

            donut = (
                alt.Chart(donut_df)
                .mark_arc(
                    innerRadius=75,
                    outerRadius=135,
                )
                .encode(
                    theta="Count:Q",
                    color=alt.Color(
                        "Level:N",
                        scale=alt.Scale(
                            domain=["LOW", "MEDIUM", "HIGH"],
                            range=["#45e0a3", "#ffcc66", "#ff5364"],
                        ),
                    ),
                    tooltip=["Level", "Count"],
                )
                .properties(height=340)
            )

            st.altair_chart(
                donut,
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            st.subheader("Traffic statistics")

            stats = pd.DataFrame(
                {
                    "Metric": [
                        "Minimum traffic",
                        "Maximum traffic",
                        "Average traffic",
                        "Median traffic",
                        "Standard deviation",
                    ],
                    "Value": [
                        float(np.min(y_test)),
                        float(np.max(y_test)),
                        float(np.mean(y_test)),
                        float(np.median(y_test)),
                        float(np.std(y_test)),
                    ],
                }
            )

            stats["Value"] = stats["Value"].round(4)

            st.dataframe(
                stats,
                hide_index=True,
                use_container_width=True,
            )

    st.write("")

    st.markdown(
        '<div class="section-title">Traffic trend</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Actual traffic intensity across test observations.'
        '</div>',
        unsafe_allow_html=True,
    )

    trend_size = min(500, len(y_test))

    trend_df = pd.DataFrame(
        {
            "Observation": np.arange(1, trend_size + 1),
            "Traffic": y_test[:trend_size],
        }
    )

    trend_chart = (
        alt.Chart(trend_df)
        .mark_line()
        .encode(
            x=alt.X("Observation:Q", title="Observation"),
            y=alt.Y("Traffic:Q", title="Traffic intensity"),
            tooltip=[
                "Observation",
                alt.Tooltip("Traffic:Q", format=".4f"),
            ],
        )
        .properties(height=350)
        .interactive()
    )

    with st.container(border=True):
        st.altair_chart(
            trend_chart,
            use_container_width=True,
        )

    st.write("")

    st.markdown(
        '<div class="section-title">Actual vs predicted</div>',
        unsafe_allow_html=True,
    )

    comparison_size = min(300, len(y_test))

    comparison_df = pd.DataFrame(
        {
            "Observation": np.arange(1, comparison_size + 1),
            "Actual": y_test[:comparison_size],
            "Predicted": predictions[:comparison_size],
        }
    )

    comparison_long = comparison_df.melt(
        id_vars="Observation",
        value_vars=["Actual", "Predicted"],
        var_name="Type",
        value_name="Traffic",
    )

    comparison_chart = (
        alt.Chart(comparison_long)
        .mark_line()
        .encode(
            x=alt.X("Observation:Q", title="Observation"),
            y=alt.Y("Traffic:Q", title="Traffic intensity"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Actual", "Predicted"],
                    range=["#ffffff", "#ff5364"],
                ),
            ),
            tooltip=[
                "Observation",
                "Type",
                alt.Tooltip("Traffic:Q", format=".4f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )

    with st.container(border=True):
        st.altair_chart(
            comparison_chart,
            use_container_width=True,
        )

    st.write("")

    average_traffic = float(np.mean(y_test))
    max_traffic = float(np.max(y_test))
    high_percentage = high / total * 100 if total else 0

    with st.container(border=True):
        st.subheader("Analytics insight")

        st.markdown(
            f"""
            **TrafficSense analyzed {total:,} test observations.**

            - Average traffic intensity: **{average_traffic:.4f}**
            - Peak traffic intensity: **{max_traffic:.4f}**
            - High-congestion observations: **{high:,} ({high_percentage:.1f}%)**
            """
        )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="page-title">Model Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Understand model performance, feature importance and individual predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">XGBoost performance</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("R² SCORE", "84.37%")

    with c2:
        st.metric("RMSE", "0.0809")

    with c3:
        st.metric("MAE", "0.0618")

    with c4:
        st.metric("TRAINING SAMPLES", f"{len(X_train):,}")

    st.write("")

    st.markdown(
        '<div class="section-title">Model comparison</div>',
        unsafe_allow_html=True,
    )

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest",
                "XGBoost",
            ],
            "R²": [0.7328, 0.8328, 0.8437],
            "RMSE": [0.1057, 0.0836, 0.0809],
            "MAE": [0.0766, 0.0602, 0.0618],
        }
    )

    st.dataframe(
        model_comparison.style.format(
            {
                "R²": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.write("")

    st.markdown(
        '<div class="section-title">Feature importance</div>',
        unsafe_allow_html=True,
    )

    importance_values = np.asarray(
        model.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    importance_df["Importance"] *= 100

    importance_chart = (
        alt.Chart(importance_df)
        .mark_bar(
            cornerRadiusTopRight=6,
            cornerRadiusBottomRight=6,
        )
        .encode(
            x=alt.X(
                "Importance:Q",
                title="Importance (%)",
            ),
            y=alt.Y(
                "Feature:N",
                sort="-x",
                title="",
            ),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip(
                    "Importance:Q",
                    format=".2f",
                ),
            ],
        )
        .properties(height=300)
    )

    with st.container(border=True):
        st.altair_chart(
            importance_chart,
            use_container_width=True,
        )

    st.write("")

    # --------------------------------------------------------
    # SHAP EXPLAINABILITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction explainability</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understand how each traffic feature influences one individual prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    shap_observation = st.slider(
        "Select observation to explain",
        min_value=1,
        max_value=len(X_test),
        value=1,
        key="shap_observation",
    )

    shap_index = shap_observation - 1
    shap_input = X_test[shap_index].reshape(1, -1)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_input)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values).flatten()

        if len(shap_values) != len(feature_names):
            st.warning(
                "SHAP returned a different number of feature values "
                "than the dashboard feature list."
            )
        else:
            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": shap_values,
                    "Feature Value": shap_input[0],
                }
            )

            shap_df["Impact"] = np.where(
                shap_df["SHAP Value"] >= 0,
                "Increases prediction",
                "Decreases prediction",
            )

            shap_df["Absolute Impact"] = shap_df[
                "SHAP Value"
            ].abs()

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False,
            )

            shap_prediction = float(
                model.predict(shap_input)[0]
            )

            shap_classification = classify_traffic(
                shap_prediction
            )

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "PREDICTED TRAFFIC",
                    f"{shap_prediction:.4f}",
                )

            with s2:
                st.metric(
                    "CONGESTION",
                    shap_classification,
                )

            with s3:
                st.metric(
                    "STRONGEST FEATURE",
                    str(shap_df.iloc[0]["Feature"]),
                )

            st.write("")

            with st.container(border=True):
                st.subheader("Feature contribution")

                shap_chart = (
                    alt.Chart(shap_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "SHAP Value:Q",
                            title="Impact on prediction",
                        ),
                        y=alt.Y(
                            "Feature:N",
                            sort="-x",
                            title="",
                        ),
                        color=alt.condition(
                            "datum['SHAP Value'] >= 0",
                            alt.value("#ff5364"),
                            alt.value("#45e0a3"),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Feature:N",
                                title="Feature",
                            ),
                            alt.Tooltip(
                                "Feature Value:Q",
                                title="Feature value",
                                format=".4f",
                            ),
                            alt.Tooltip(
                                "SHAP Value:Q",
                                title="SHAP impact",
                                format=".4f",
                            ),
                            alt.Tooltip(
                                "Impact:N",
                                title="Effect",
                            ),
                        ],
                    )
                    .properties(height=320)
                )

                st.altair_chart(
                    shap_chart,
                    use_container_width=True,
                )

            st.write("")

            with st.container(border=True):
                st.subheader("Prediction breakdown")

                display_df = shap_df[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value",
                        "Impact",
                    ]
                ].copy()

                display_df["Feature Value"] = display_df[
                    "Feature Value"
                ].round(4)

                display_df["SHAP Value"] = display_df[
                    "SHAP Value"
                ].round(4)

                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                )

            st.write("")

            top_feature = shap_df.iloc[0]

            with st.container(border=True):
                st.subheader("AI explanation")

                if float(top_feature["SHAP Value"]) >= 0:
                    direction_text = "increases"
                else:
                    direction_text = "decreases"

                st.markdown(
                    f"""
                    The model predicts a traffic intensity of
                    **{shap_prediction:.4f}**, classified as
                    **{shap_classification} congestion**.

                    The strongest contributing feature is
                    **{top_feature["Feature"]}**, with a value of
                    **{float(top_feature["Feature Value"]):.4f}**.

                    Based on its SHAP value, this feature
                    **{direction_text}** the prediction by approximately
                    **{abs(float(top_feature["SHAP Value"])):.4f}**
                    relative to the model baseline.
                    """
                )

    except Exception as exc:
        st.error(
            "SHAP explanation could not be generated for this model."
        )
        st.caption(str(exc))

    st.write("")

    st.markdown(
        '<div class="section-title">How the model works</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            **TrafficSense uses XGBoost Regression** to estimate
            traffic intensity from engineered historical traffic features.

            The prediction is then converted into three operational
            congestion levels:

            🟢 **LOW** — below 0.1971

            🟡 **MEDIUM** — from 0.1971 to below 0.4040

            🔴 **HIGH** — 0.4040 and above
            """
        )

    st.write("")

    with st.container(border=True):
        st.subheader("Why XGBoost?")

        st.markdown(
            """
            XGBoost achieved the strongest overall performance among
            the evaluated models.

            **R²:** 0.8437  
            **RMSE:** 0.0809  
            **MAE:** 0.0618

            This indicates that the model captures a substantial
            portion of the variation in traffic intensity while
            maintaining relatively low prediction error.
            """
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="page-title">About TrafficSense</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'An AI-powered traffic intelligence system for forecasting traffic intensity '
        'and identifying congestion levels.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("TrafficSense")

        st.markdown(
            """
            **TrafficSense** is a machine-learning-based traffic
            forecasting system designed to transform historical
            traffic observations into understandable congestion insights.

            The system uses engineered traffic features and an
            **XGBoost regression model** to predict traffic intensity.

            Predictions are then classified into:

            🟢 **LOW** — relatively clear traffic

            🟡 **MEDIUM** — moderate congestion

            🔴 **HIGH** — heavy congestion
            """
        )

    st.write("")

    st.markdown(
        '<div class="section-title">Technology stack</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("LANGUAGE", "Python")

    with c2:
        st.metric("ML MODEL", "XGBoost")

    with c3:
        st.metric("DASHBOARD", "Streamlit")

    with c4:
        st.metric("DATA", "NumPy / Pandas")

    st.write("")

    st.markdown(
        '<div class="section-title">Prediction pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "01",
            "Historical traffic data",
            "Traffic observations are loaded from the dataset.",
        ),
        (
            "02",
            "Feature engineering",
            "Historical observations are transformed into predictive features.",
        ),
        (
            "03",
            "XGBoost prediction",
            "The trained model estimates traffic intensity.",
        ),
        (
            "04",
            "Congestion classification",
            "Predictions are converted into LOW, MEDIUM or HIGH.",
        ),
        (
            "05",
            "Dashboard insights",
            "Results are presented through the interactive dashboard.",
        ),
    ]

    for number, title, description in steps:
        with st.container(border=True):
            col1, col2 = st.columns([0.12, 0.88])

            with col1:
                st.markdown(f"### {number}")

            with col2:
                st.markdown(f"**{title}**")
                st.caption(description)

    st.write("")

    st.markdown(
        '<div class="section-title">Project metrics</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("TRAINING SAMPLES", f"{len(X_train):,}")

    with c2:
        st.metric("TEST SAMPLES", f"{len(X_test):,}")

    with c3:
        st.metric("MODEL R²", "84.37%")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-custom">
        <b>TrafficSense</b><br>
        AI Traffic Intelligence<br><br>
        Built with Python · XGBoost · Streamlit · Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
