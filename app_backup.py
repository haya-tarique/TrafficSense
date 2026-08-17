import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrafficSense",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM DARK THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 5%, rgba(255, 65, 85, 0.08), transparent 25%),
        radial-gradient(circle at 20% 80%, rgba(80, 110, 255, 0.05), transparent 30%),
        #070b11;
    color: #f5f7fa;
}

/* Main content */

.block-container {
    max-width: 1450px;
    padding: 2rem 3rem 4rem 3rem;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #090e15;
    border-right: 1px solid #202833;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.4rem;
}

/* Sidebar text */

.sidebar-title {
    font-size: 27px;
    font-weight: 800;
    color: white;
}

.sidebar-title span {
    color: #ff5364;
}

.sidebar-subtitle {
    color: #718096;
    font-size: 13px;
    margin-top: 5px;
    margin-bottom: 25px;
}

/* Navigation */

.nav-label {
    color: #66758a;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-top: 25px;
    margin-bottom: 8px;
}

/* Radio buttons */

div[data-testid="stRadio"] > label {
    color: #738196 !important;
    font-weight: 600;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 5px;
}

div[data-testid="stRadio"] label {
    padding: 10px 12px;
    border-radius: 10px;
    transition: 0.2s;
}

div[data-testid="stRadio"] label:hover {
    background: #141b25;
    color: white !important;
}

/* Top bar */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 25px;
    border-bottom: 1px solid #202833;
    margin-bottom: 30px;
}

.topbar-small {
    color: #718096;
    font-size: 13px;
}

.online {
    color: #45e0a3;
    font-size: 13px;
    font-weight: 700;
}

/* Headings */

.page-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.page-description {
    color: #718096;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Native Streamlit metric cards */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #101722,
        #0b1018
    );
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

/* Containers */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(
        145deg,
        rgba(18,25,35,0.95),
        rgba(9,14,21,0.95)
    );
    border: 1px solid #202a36;
    border-radius: 18px;
}

/* Buttons */

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

/* Slider */

div[data-testid="stSlider"] [role="slider"] {
    background-color: #ff5364;
}

/* Progress */

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #45e0a3, #ffcc66, #ff5364);
}

/* Tables */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Footer */

.footer {
    text-align: center;
    color: #4f5d70;
    font-size: 12px;
    padding: 45px 0 15px;
}

</style>
""", unsafe_allow_html=True)


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
            "Model not found at:\n\n"
            "models/xgboost_traffic_model.pkl"
        )

        st.stop()

    return joblib.load(model_path)


# ============================================================
# CONGESTION
# ============================================================

def classify_traffic(value):

    if value < 0.1971:
        return "LOW"

    elif value < 0.4040:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# LOAD
# ============================================================

X_train, y_train, X_test, y_test = load_data()

model = load_model()

predictions = model.predict(X_test)

predictions = np.asarray(predictions)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🚦 Traffic<span>Sense</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI-powered traffic intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Traffic Forecast",
            "Analytics",
            "Model Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">FORECAST CONTROL</div>',
        unsafe_allow_html=True
    )

    observation = st.slider(
        "Traffic observation",
        1,
        len(X_test),
        1
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">SYSTEM STATUS</div>',
        unsafe_allow_html=True
    )

    st.success("● MODEL ONLINE")

    st.caption("XGBoost prediction engine is active")


# ============================================================
# SELECTED OBSERVATION
# ============================================================

index = observation - 1

selected_features = X_test[index].reshape(1, -1)

prediction = float(model.predict(selected_features)[0])

actual = float(y_test[index])

difference = prediction - actual

classification = classify_traffic(prediction)


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div>
            <b>TrafficSense Intelligence Platform</b>
        </div>
        <div>
            <span class="online">● SYSTEM ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="page-title">Traffic intelligence at a glance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Monitor traffic conditions, model performance and congestion trends.'
        '</div>',
        unsafe_allow_html=True
    )

    # KPI ROW

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² SCORE",
            "84.37%",
            "Model accuracy"
        )

    with col2:
        st.metric(
            "RMSE",
            "0.0809",
            "Prediction error"
        )

    with col3:
        st.metric(
            "MAE",
            "0.0618",
            "Average error"
        )

    with col4:
        st.metric(
            "TEST OBSERVATIONS",
            f"{len(X_test):,}",
            "Traffic records"
        )

    st.write("")

    # Current traffic

    left, right = st.columns([1.4, 1])

    with left:

        with st.container(border=True):

            st.subheader("Current traffic forecast")

            st.caption(
                f"Observation #{observation:,} from the test dataset"
            )

            st.metric(
                "Predicted intensity",
                f"{prediction:.4f}"
            )

            if classification == "LOW":
                st.success("🟢 LOW CONGESTION")

            elif classification == "MEDIUM":
                st.warning("🟡 MEDIUM CONGESTION")

            else:
                st.error("🔴 HIGH CONGESTION")

            st.progress(
                min(max(prediction, 0.0), 1.0)
            )

            c1, c2 = st.columns(2)

            with c1:
                st.caption("Actual")
                st.write(f"### {actual:.4f}")

            with c2:
                st.caption("Prediction difference")
                st.write(f"### {difference:+.4f}")

    with right:

        with st.container(border=True):

            st.subheader("Traffic signals")

            feature_names = [
                "Last Traffic Value",
                "Average Traffic",
                "Traffic Variation",
                "Minimum Traffic",
                "Maximum Traffic"
            ]

            feature_values = selected_features[0]

            feature_df = pd.DataFrame({
                "Signal": feature_names,
                "Value": [
                    round(float(x), 4)
                    for x in feature_values[:5]
                ]
            })

            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True
            )

    st.write("")

    # Trend

    with st.container(border=True):

        st.subheader("Traffic activity")

        chart_df = pd.DataFrame({
            "Observation": np.arange(
                1,
                min(501, len(y_test) + 1)
            ),
            "Traffic": y_test[:500]
        })

        chart = (
            alt.Chart(chart_df)
            .mark_area(
                line=True,
                opacity=0.25
            )
            .encode(
                x=alt.X(
                    "Observation:Q",
                    title="Observation"
                ),
                y=alt.Y(
                    "Traffic:Q",
                    title="Traffic intensity"
                ),
                tooltip=[
                    "Observation",
                    "Traffic"
                ]
            )
            .properties(height=350)
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )


# ============================================================
# TRAFFIC FORECAST
# ============================================================

elif page == "Traffic Forecast":

    st.markdown(
        '<div class="page-title">Traffic forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Explore individual traffic observations and AI predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.slider(
        "Select observation",
        1,
        len(X_test),
        observation,
        key="forecast_slider"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PREDICTED",
            f"{prediction:.4f}"
        )

    with col2:
        st.metric(
            "ACTUAL",
            f"{actual:.4f}"
        )

    with col3:
        st.metric(
            "ERROR",
            f"{abs(difference):.4f}"
        )

    st.write("")

    with st.container(border=True):

        st.subheader("Congestion status")

        if classification == "LOW":
            st.success(
                "🟢 LOW — Traffic conditions are relatively clear."
            )

        elif classification == "MEDIUM":
            st.warning(
                "🟡 MEDIUM — Moderate congestion detected."
            )

        else:
            st.error(
                "🔴 HIGH — Heavy congestion detected."
            )

        st.progress(
            min(max(prediction, 0), 1)
        )

    st.write("")

    with st.container(border=True):

        st.subheader("Model input signals")

        feature_names = [
            "Last Traffic Value",
            "Average Traffic",
            "Traffic Variation",
            "Minimum Traffic",
            "Maximum Traffic"
        ]

        values = selected_features[0]

        for i, name in enumerate(feature_names):

            if i < len(values):

                st.write(
                    f"**{name}** — `{float(values[i]):.4f}`"
                )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="page-title">Traffic analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Understand congestion patterns across the test dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    classes = [
        classify_traffic(float(x))
        for x in predictions
    ]

    counts = pd.Series(classes).value_counts()

    low = int(counts.get("LOW", 0))
    medium = int(counts.get("MEDIUM", 0))
    high = int(counts.get("HIGH", 0))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("LOW", f"{low:,}")

    with c2:
        st.metric("MEDIUM", f"{medium:,}")

    with c3:
        st.metric("HIGH", f"{high:,}")

    st.write("")

    left, right = st.columns(2)

    with left:

        with st.container(border=True):

            st.subheader("Congestion distribution")

            donut_df = pd.DataFrame({
                "Level": [
                    "LOW",
                    "MEDIUM",
                    "HIGH"
                ],
                "Count": [
                    low,
                    medium,
                    high
                ]
            })

            donut = (
                alt.Chart(donut_df)
                .mark_arc(
                    innerRadius=70,
                    outerRadius=130
                )
                .encode(
                    theta="Count:Q",
                    color=alt.Color(
                        "Level:N",
                        scale=alt.Scale(
                            domain=[
                                "LOW",
                                "MEDIUM",
                                "HIGH"
                            ],
                            range=[
                                "#45e0a3",
                                "#ffcc66",
                                "#ff5364"
                            ]
                        )
                    ),
                    tooltip=[
                        "Level",
                        "Count"
                    ]
                )
                .properties(height=330)
            )

            st.altair_chart(
                donut,
                use_container_width=True
            )

    with right:

        with st.container(border=True):

            st.subheader("Traffic statistics")

            stats = pd.DataFrame({
                "Metric": [
                    "Minimum",
                    "Maximum",
                    "Average",
                    "Median",
                    "Std. deviation"
                ],
                "Value": [
                    float(np.min(y_test)),
                    float(np.max(y_test)),
                    float(np.mean(y_test)),
                    float(np.median(y_test)),
                    float(np.std(y_test))
                ]
            })

            stats["Value"] = stats["Value"].round(4)

            st.dataframe(
                stats,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="page-title">Model insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Understand model performance and the signals used for forecasting.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R²",
            "84.37%"
        )

    with col2:
        st.metric(
            "RMSE",
            "0.0809"
        )

    with col3:
        st.metric(
            "MAE",
            "0.0618"
        )

    st.write("")

    with st.container(border=True):

        st.subheader("Feature importance")

        feature_names = [
            "Last Traffic Value",
            "Average Traffic",
            "Traffic Variation",
            "Minimum Traffic",
            "Maximum Traffic"
        ]

        # Use first five model features where available
        importance_values = np.ones(
            min(5, X_test.shape[1])
        )

        try:

            if hasattr(model, "feature_importances_"):

                importance_values = np.asarray(
                    model.feature_importances_
                )[:5]

        except Exception:
            pass

        importance_df = pd.DataFrame({
            "Feature": feature_names[:len(importance_values)],
            "Importance": importance_values
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        importance_chart = (
            alt.Chart(importance_df)
            .mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6
            )
            .encode(
                x=alt.X(
                    "Importance:Q",
                    title="Importance"
                ),
                y=alt.Y(
                    "Feature:N",
                    sort="-x",
                    title=""
                ),
                tooltip=[
                    "Feature",
                    "Importance"
                ]
            )
            .properties(height=300)
        )

        st.altair_chart(
            importance_chart,
            use_container_width=True
        )

    st.write("")

    with st.container(border=True):

        st.subheader("How TrafficSense works")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### 01")

            st.write("**Historical data**")

            st.caption(
                "Historical traffic observations are "
                "prepared for machine learning."
            )

        with col2:

            st.markdown("### 02")

            st.write("**Feature engineering**")

            st.caption(
                "Traffic history is transformed into "
                "meaningful predictive signals."
            )

        with col3:

            st.markdown("### 03")

            st.write("**AI prediction**")

            st.caption(
                "XGBoost predicts traffic intensity "
                "and congestion level."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TrafficSense AI · Intelligent Traffic Forecasting
        <br>
        Python · Streamlit · XGBoost · Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)