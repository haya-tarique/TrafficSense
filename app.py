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
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    * {
        font-family: Arial, sans-serif;
    }

    .stApp {
        background-color: #070b11;
        color: #f5f7fa;
    }

    .block-container {
        max-width: 1450px;
        padding: 2rem 3rem 4rem 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #090e15;
        border-right: 1px solid #202833;
    }

    .page-title {
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .page-description {
        color: #8a96a8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: white;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #8a96a8;
        font-size: 13px;
        margin-bottom: 15px;
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
            "data/processed_features.npz was not found."
        )
        st.stop()

    data = np.load(feature_file)

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

    model_path = Path(
        "models/xgboost_traffic_model.pkl"
    )

    if not model_path.exists():
        st.error(
            "models/xgboost_traffic_model.pkl was not found."
        )
        st.stop()

    return joblib.load(model_path)


# ============================================================
# LOAD
# ============================================================

X_train, y_train, X_test, y_test = load_data()

model = load_model()


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "Last Traffic Value",
    "Average Traffic",
    "Maximum Traffic",
    "Minimum Traffic",
    "Traffic Variation",
]


# ============================================================
# PREPARE DATA FOR DASHBOARD
# ============================================================

def prepare_features(X):

    """
    Converts the available processed data into:

        observations × nodes × 5 features

    Supports both:

        1. New format:
           (observations * 36, 5)

        2. Older format:
           (observations, 36, 48)
    """

    X = np.asarray(X)

    # --------------------------------------------------------
    # CASE 1
    # Already engineered:
    # (observations * 36, 5)
    # --------------------------------------------------------

    if X.ndim == 2 and X.shape[1] == 5:

        total_rows = X.shape[0]

        if total_rows % 36 != 0:
            raise ValueError(
                "The number of rows in X is not divisible by 36 nodes."
            )

        observations = total_rows // 36

        return X.reshape(
            observations,
            36,
            5
        )

    # --------------------------------------------------------
    # CASE 2
    # Raw historical format:
    # (observations, 36, 48)
    # --------------------------------------------------------

    if X.ndim == 3:

        observations = X.shape[0]
        nodes = X.shape[1]

        if nodes != 36:
            raise ValueError(
                f"Expected 36 nodes but found {nodes}."
            )

        last_value = X[:, :, -1]

        mean_value = X.mean(axis=2)

        max_value = X.max(axis=2)

        min_value = X.min(axis=2)

        std_value = X.std(axis=2)

        features = np.stack(
            [
                last_value,
                mean_value,
                max_value,
                min_value,
                std_value,
            ],
            axis=2,
        )

        return features

    raise ValueError(
        f"Unsupported X shape: {X.shape}"
    )


# Prepare test features

X_test_nodes = prepare_features(X_test)


# Prepare training features if possible

try:

    X_train_nodes = prepare_features(X_train)

except Exception:

    X_train_nodes = None


# ============================================================
# PREPARE TARGETS
# ============================================================

def prepare_targets(y):

    """
    Converts targets into:

        observations × nodes
    """

    y = np.asarray(y)

    # Already:
    # observations × nodes

    if y.ndim == 2:

        if y.shape[1] == 36:
            return y

        if y.shape[0] == 36:
            return y.T

    # Flattened:
    # observations * nodes

    if y.ndim == 1:

        if len(y) % 36 != 0:
            raise ValueError(
                "Target size is not divisible by 36."
            )

        return y.reshape(
            -1,
            36
        )

    raise ValueError(
        f"Unsupported target shape: {y.shape}"
    )


y_test_nodes = prepare_targets(y_test)

y_train_nodes = prepare_targets(y_train)


# ============================================================
# CHECK DATA CONSISTENCY
# ============================================================

number_of_observations = min(
    X_test_nodes.shape[0],
    y_test_nodes.shape[0],
)

number_of_nodes = 36


# ============================================================
# CONGESTION CLASSIFICATION
# ============================================================

def classify_traffic(value):

    if value < 0.1971:
        return "LOW"

    elif value < 0.4040:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# PREDICT ONE NODE
# ============================================================

def predict_node(
    observation_index,
    node_index
):

    features = X_test_nodes[
        observation_index,
        node_index
    ]

    features_for_model = features.reshape(
        1,
        -1
    )

    prediction = float(
        model.predict(
            features_for_model
        )[0]
    )

    actual = float(
        y_test_nodes[
            observation_index,
            node_index
        ]
    )

    return (
        features,
        prediction,
        actual
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="color:white;">
        Traffic<span style="color:#ff5364;">Sense</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "AI-powered traffic intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Traffic Forecast",
            "Analytics",
            "Model Insights",
            "About",
        ],
    )

    st.divider()

    st.subheader("System Status")

    st.success(
        "MODEL ONLINE"
    )

    st.caption(
        "XGBoost prediction engine is active"
    )

    st.divider()

    st.caption(
        f"Traffic Nodes: {number_of_nodes}"
    )

    st.caption(
        f"Test Observations: {number_of_observations:,}"
    )


# ============================================================
# GLOBAL SELECTION
# ============================================================

if "global_observation" not in st.session_state:

    st.session_state.global_observation = 1


if "global_node" not in st.session_state:

    st.session_state.global_node = 1


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="page-title">'
        'Traffic Intelligence at a Glance'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Monitor traffic conditions, congestion levels, '
        'and machine-learning predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "R² SCORE",
            "84.37%"
        )

    with c2:

        st.metric(
            "RMSE",
            "0.0809"
        )

    with c3:

        st.metric(
            "MAE",
            "0.0618"
        )

    with c4:

        st.metric(
            "TEST OBSERVATIONS",
            f"{number_of_observations:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # CURRENT NODE
    # --------------------------------------------------------

    observation = (
        st.session_state.global_observation
    )

    node = (
        st.session_state.global_node
    )

    features, prediction, actual = predict_node(
        observation - 1,
        node - 1
    )

    classification = classify_traffic(
        prediction
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Current Traffic Forecast"
        )

        st.caption(
            f"Observation #{observation:,} "
            f"• Node {node}"
        )

        st.metric(
            "Predicted Traffic",
            f"{prediction:.4f}"
        )

        if classification == "LOW":

            st.success(
                "🟢 LOW CONGESTION"
            )

        elif classification == "MEDIUM":

            st.warning(
                "🟡 MEDIUM CONGESTION"
            )

        else:

            st.error(
                "🔴 HIGH CONGESTION"
            )

        st.progress(
            min(
                max(prediction, 0.0),
                1.0
            )
        )

    with right:

        st.subheader(
            "Traffic Signals"
        )

        signal_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Value": [
                    round(float(v), 4)
                    for v in features
                ],
            }
        )

        st.dataframe(
            signal_df,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # TRAFFIC TREND
    # --------------------------------------------------------

    st.subheader(
        "Traffic Activity"
    )

    trend_size = min(
        500,
        number_of_observations
    )

    trend_values = np.mean(
        y_test_nodes[:trend_size],
        axis=1
    )

    trend_df = pd.DataFrame(
        {
            "Observation": np.arange(
                1,
                trend_size + 1
            ),
            "Traffic": trend_values,
        }
    )

    chart = (
        alt.Chart(trend_df)
        .mark_line()
        .encode(
            x=alt.X(
                "Observation:Q",
                title="Observation"
            ),
            y=alt.Y(
                "Traffic:Q",
                title="Average Traffic"
            ),
            tooltip=[
                "Observation",
                alt.Tooltip(
                    "Traffic:Q",
                    format=".4f"
                ),
            ],
        )
        .properties(
            height=350
        )
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
        '<div class="page-title">'
        'Traffic Forecast'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Select a traffic observation and node to generate '
        'an AI-powered traffic forecast.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    observation = st.slider(
        "Select Traffic Observation",
        min_value=1,
        max_value=number_of_observations,
        value=st.session_state.global_observation,
        key="forecast_observation",
    )

    st.session_state.global_observation = (
        observation
    )

    # --------------------------------------------------------
    # NODE
    # --------------------------------------------------------

    node = st.selectbox(
        "Select Traffic Node / Location",
        options=list(
            range(1, number_of_nodes + 1)
        ),
        index=(
            st.session_state.global_node - 1
        ),
        format_func=lambda x:
            f"Node {x}",
        key="forecast_node",
    )

    st.session_state.global_node = node

    # --------------------------------------------------------
    # GET PREDICTION
    # --------------------------------------------------------

    features, prediction, actual = predict_node(
        observation - 1,
        node - 1
    )

    difference = (
        prediction - actual
    )

    classification = classify_traffic(
        prediction
    )

    st.divider()

    # --------------------------------------------------------
    # FORECAST RESULT
    # --------------------------------------------------------

    left, right = st.columns(
        [1.3, 1]
    )

    with left:

        st.subheader(
            "Predicted Traffic Intensity"
        )

        st.metric(
            "Prediction",
            f"{prediction:.4f}"
        )

        if classification == "LOW":

            st.success(
                "🟢 LOW CONGESTION — "
                "Traffic is relatively clear."
            )

        elif classification == "MEDIUM":

            st.warning(
                "🟡 MEDIUM CONGESTION — "
                "Moderate congestion expected."
            )

        else:

            st.error(
                "🔴 HIGH CONGESTION — "
                "Heavy congestion expected."
            )

        st.progress(
            min(
                max(prediction, 0.0),
                1.0
            )
        )

    with right:

        st.subheader(
            "Prediction Analysis"
        )

        a, b = st.columns(2)

        with a:

            st.metric(
                "Predicted",
                f"{prediction:.4f}"
            )

        with b:

            st.metric(
                "Actual",
                f"{actual:.4f}"
            )

        st.metric(
            "Absolute Error",
            f"{abs(difference):.4f}"
        )

        if difference > 0:

            st.caption(
                "Prediction is above the actual value."
            )

        elif difference < 0:

            st.caption(
                "Prediction is below the actual value."
            )

        else:

            st.caption(
                "Prediction exactly matches the actual value."
            )

    st.divider()

    # ========================================================
    # SELECTED LOCATION
    # ========================================================

    st.subheader(
        "Selected Traffic Node"
    )

    location_col1, location_col2 = st.columns(2)

    with location_col1:

        st.metric(
            "Node",
            f"Node {node}"
        )

    with location_col2:

        st.metric(
            "Observation",
            f"#{observation}"
        )

    st.info(
        f"📍 Traffic location selected: Node {node}"
    )

    st.divider()

    # ========================================================
    # MODEL INPUT SIGNALS
    # ========================================================

    st.subheader(
        "Model Input Signals"
    )

    st.caption(
        "Five engineered traffic features supplied "
        "to the XGBoost model."
    )

    feature_col1, feature_col2, feature_col3, feature_col4, feature_col5 = st.columns(5)

    with feature_col1:

        st.metric(
            "Last Traffic",
            f"{features[0]:.4f}"
        )

    with feature_col2:

        st.metric(
            "Average Traffic",
            f"{features[1]:.4f}"
        )

    with feature_col3:

        st.metric(
            "Maximum Traffic",
            f"{features[2]:.4f}"
        )

    with feature_col4:

        st.metric(
            "Minimum Traffic",
            f"{features[3]:.4f}"
        )

    with feature_col5:

        st.metric(
            "Traffic Variation",
            f"{features[4]:.4f}"
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE TABLE
    # --------------------------------------------------------

    st.subheader(
        "Detailed Model Inputs"
    )

    input_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": [
                round(
                    float(value),
                    4
                )
                for value in features
            ],
        }
    )

    st.dataframe(
        input_df,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "AI Forecast Interpretation"
    )

    st.write(
        f"The XGBoost model predicts a traffic "
        f"intensity of **{prediction:.4f}** "
        f"for **Node {node}** at "
        f"**Observation #{observation}**."
    )

    st.write(
        f"The predicted traffic level is classified "
        f"as **{classification} congestion**."
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="page-title">'
        'Traffic Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Explore traffic distribution, statistics, '
        'and prediction performance.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PREDICT ALL TEST DATA
    # --------------------------------------------------------

    flat_X_test = X_test_nodes.reshape(
        -1,
        5
    )

    predictions_flat = np.asarray(
        model.predict(flat_X_test)
    )

    predictions_nodes = predictions_flat.reshape(
        number_of_observations,
        number_of_nodes
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classes = [
        classify_traffic(float(value))
        for value in predictions_flat
    ]

    counts = pd.Series(
        classes
    ).value_counts()

    low = int(
        counts.get(
            "LOW",
            0
        )
    )

    medium = int(
        counts.get(
            "MEDIUM",
            0
        )
    )

    high = int(
        counts.get(
            "HIGH",
            0
        )
    )

    total = (
        low +
        medium +
        high
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "TOTAL",
            f"{total:,}"
        )

    with c2:

        st.metric(
            "LOW",
            f"{low:,}"
        )

    with c3:

        st.metric(
            "MEDIUM",
            f"{medium:,}"
        )

    with c4:

        st.metric(
            "HIGH",
            f"{high:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Congestion Distribution"
        )

        donut_df = pd.DataFrame(
            {
                "Level": [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                ],
                "Count": [
                    low,
                    medium,
                    high,
                ],
            }
        )

        donut = (
            alt.Chart(donut_df)
            .mark_arc(
                innerRadius=70
            )
            .encode(
                theta="Count:Q",
                color="Level:N",
                tooltip=[
                    "Level",
                    "Count"
                ],
            )
            .properties(
                height=350
            )
        )

        st.altair_chart(
            donut,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Traffic Statistics"
        )

        stats = pd.DataFrame(
            {
                "Metric": [
                    "Minimum",
                    "Maximum",
                    "Average",
                    "Median",
                    "Standard Deviation",
                ],
                "Value": [
                    float(
                        np.min(
                            y_test_nodes
                        )
                    ),
                    float(
                        np.max(
                            y_test_nodes
                        )
                    ),
                    float(
                        np.mean(
                            y_test_nodes
                        )
                    ),
                    float(
                        np.median(
                            y_test_nodes
                        )
                    ),
                    float(
                        np.std(
                            y_test_nodes
                        )
                    ),
                ],
            }
        )

        stats["Value"] = (
            stats["Value"]
            .round(4)
        )

        st.dataframe(
            stats,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # TRAFFIC TREND
    # --------------------------------------------------------

    st.subheader(
        "Traffic Trend"
    )

    trend_size = min(
        500,
        number_of_observations
    )

    actual_average = np.mean(
        y_test_nodes[:trend_size],
        axis=1
    )

    predicted_average = np.mean(
        predictions_nodes[:trend_size],
        axis=1
    )

    trend_df = pd.DataFrame(
        {
            "Observation": np.arange(
                1,
                trend_size + 1
            ),
            "Actual": actual_average,
            "Predicted": predicted_average,
        }
    )

    trend_long = trend_df.melt(
        id_vars="Observation",
        value_vars=[
            "Actual",
            "Predicted",
        ],
        var_name="Type",
        value_name="Traffic",
    )

    chart = (
        alt.Chart(trend_long)
        .mark_line()
        .encode(
            x=alt.X(
                "Observation:Q",
                title="Observation"
            ),
            y=alt.Y(
                "Traffic:Q",
                title="Traffic Intensity"
            ),
            color="Type:N",
            tooltip=[
                "Observation",
                "Type",
                alt.Tooltip(
                    "Traffic:Q",
                    format=".4f"
                ),
            ],
        )
        .properties(
            height=400
        )
        .interactive()
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # INSIGHT
    # --------------------------------------------------------

    average_traffic = float(
        np.mean(y_test_nodes)
    )

    max_traffic = float(
        np.max(y_test_nodes)
    )

    high_percentage = (
        high / total * 100
        if total > 0
        else 0
    )

    st.subheader(
        "Analytics Insight"
    )

    st.write(
        f"TrafficSense analyzed "
        f"**{total:,} node observations**."
    )

    st.write(
        f"Average traffic intensity: "
        f"**{average_traffic:.4f}**"
    )

    st.write(
        f"Peak traffic intensity: "
        f"**{max_traffic:.4f}**"
    )

    st.write(
        f"High-congestion observations: "
        f"**{high:,} ({high_percentage:.1f}%)**"
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="page-title">'
        'Model Insights'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Understand model performance and feature importance.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "R² SCORE",
            "84.37%"
        )

    with c2:

        st.metric(
            "RMSE",
            "0.0809"
        )

    with c3:

        st.metric(
            "MAE",
            "0.0618"
        )

    with c4:

        st.metric(
            "TRAINING SAMPLES",
            f"{len(y_train_nodes):,}"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest",
                "XGBoost",
            ],
            "R²": [
                0.7328,
                0.8328,
                0.8437,
            ],
            "RMSE": [
                0.1057,
                0.0836,
                0.0809,
            ],
            "MAE": [
                0.0766,
                0.0602,
                0.0618,
            ],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )

    importance = np.asarray(
        model.feature_importances_
    )

    if len(importance) == 5:

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )

        importance_df[
            "Importance"
        ] *= 100

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        importance_chart = (
            alt.Chart(
                importance_df
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "Importance:Q",
                    title="Importance (%)"
                ),
                y=alt.Y(
                    "Feature:N",
                    sort="-x"
                ),
                tooltip=[
                    "Feature",
                    alt.Tooltip(
                        "Importance:Q",
                        format=".2f"
                    ),
                ],
            )
            .properties(
                height=350
            )
        )

        st.altair_chart(
            importance_chart,
            use_container_width=True,
        )

    else:

        st.warning(
            "The loaded model does not contain "
            "five feature importance values."
        )

    st.divider()

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader(
        "Prediction Explainability"
    )

    shap_observation = st.slider(
        "Select Observation",
        min_value=1,
        max_value=number_of_observations,
        value=1,
        key="shap_observation",
    )

    shap_node = st.selectbox(
        "Select Node",
        options=list(
            range(
                1,
                number_of_nodes + 1
            )
        ),
        format_func=lambda x:
            f"Node {x}",
        key="shap_node",
    )

    shap_features = X_test_nodes[
        shap_observation - 1,
        shap_node - 1
    ].reshape(
        1,
        -1
    )

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            shap_features
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        ).flatten()

        if len(shap_values) != 5:

            st.warning(
                "SHAP returned an unexpected "
                "number of feature values."
            )

        else:

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": shap_values,
                    "Feature Value": shap_features[0],
                }
            )

            shap_df[
                "Absolute Impact"
            ] = shap_df[
                "SHAP Value"
            ].abs()

            shap_df = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
            )

            shap_prediction = float(
                model.predict(
                    shap_features
                )[0]
            )

            shap_class = classify_traffic(
                shap_prediction
            )

            st.metric(
                "Predicted Traffic",
                f"{shap_prediction:.4f}"
            )

            st.info(
                f"Node {shap_node} — "
                f"{shap_class} congestion"
            )

            shap_chart = (
                alt.Chart(
                    shap_df
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "SHAP Value:Q",
                        title="SHAP Impact"
                    ),
                    y=alt.Y(
                        "Feature:N",
                        sort="-x"
                    ),
                    color="Impact:N",
                    tooltip=[
                        "Feature",
                        alt.Tooltip(
                            "Feature Value:Q",
                            format=".4f"
                        ),
                        alt.Tooltip(
                            "SHAP Value:Q",
                            format=".4f"
                        ),
                    ],
                )
                .transform_calculate(
                    Impact="""
                    datum['SHAP Value'] >= 0
                    ? 'Increases prediction'
                    : 'Decreases prediction'
                    """
                )
                .properties(
                    height=350
                )
            )

            st.altair_chart(
                shap_chart,
                use_container_width=True,
            )

            display_df = shap_df[
                [
                    "Feature",
                    "Feature Value",
                    "SHAP Value",
                ]
            ].copy()

            display_df[
                "Feature Value"
            ] = display_df[
                "Feature Value"
            ].round(4)

            display_df[
                "SHAP Value"
            ] = display_df[
                "SHAP Value"
            ].round(4)

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
            )

            strongest = shap_df.iloc[0]

            st.write(
                f"The strongest contributing feature is "
                f"**{strongest['Feature']}**."
            )

    except Exception as exc:

        st.warning(
            "SHAP explanation could not be generated."
        )

        st.caption(
            str(exc)
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL DESCRIPTION
    # --------------------------------------------------------

    st.subheader(
        "How the Model Works"
    )

    st.write(
        "TrafficSense uses an XGBoost regression model "
        "to estimate traffic intensity from engineered "
        "historical traffic features."
    )

    st.write(
        "The predicted intensity is converted into "
        "three congestion levels:"
    )

    st.write(
        "🟢 LOW — below 0.1971"
    )

    st.write(
        "🟡 MEDIUM — 0.1971 to below 0.4040"
    )

    st.write(
        "🔴 HIGH — 0.4040 and above"
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="page-title">'
        'About TrafficSense'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'AI-powered traffic forecasting and congestion analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "TrafficSense"
    )

    st.write(
        "TrafficSense is a machine-learning-based "
        "traffic forecasting system designed to "
        "transform historical traffic observations "
        "into understandable congestion insights."
    )

    st.write(
        "The system uses engineered traffic features "
        "and an XGBoost regression model to predict "
        "traffic intensity."
    )

    st.divider()

    st.subheader(
        "Technology Stack"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "LANGUAGE",
            "Python"
        )

    with c2:

        st.metric(
            "ML MODEL",
            "XGBoost"
        )

    with c3:

        st.metric(
            "DASHBOARD",
            "Streamlit"
        )

    with c4:

        st.metric(
            "DATA",
            "NumPy / Pandas"
        )

    st.divider()

    st.subheader(
        "Prediction Pipeline"
    )

    steps = [
        (
            "01",
            "Historical Traffic Data",
            "Traffic observations are loaded from the dataset.",
        ),
        (
            "02",
            "Feature Engineering",
            "Historical traffic is converted into five predictive features.",
        ),
        (
            "03",
            "XGBoost Prediction",
            "The trained model estimates traffic intensity.",
        ),
        (
            "04",
            "Congestion Classification",
            "Predictions are converted into LOW, MEDIUM or HIGH.",
        ),
        (
            "05",
            "Dashboard Insights",
            "Results are displayed through the interactive dashboard.",
        ),
    ]

    for number, title, description in steps:

        st.markdown(
            f"### {number}. {title}"
        )

        st.caption(
            description
        )

    st.divider()

    st.subheader(
        "Project Metrics"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "TRAINING OBSERVATIONS",
            f"{len(y_train_nodes):,}"
        )

    with c2:

        st.metric(
            "TEST OBSERVATIONS",
            f"{len(y_test_nodes):,}"
        )

    with c3:

        st.metric(
            "TRAFFIC NODES",
            "36"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TrafficSense • AI Traffic Intelligence • "
    "Python • XGBoost • Streamlit"
)