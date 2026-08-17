import streamlit as st
import numpy as np
import joblib
import pandas as pd


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="TrafficSense",
    page_icon="🚦",
    layout="wide"
)


# -------------------------------
# Load model and data
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/xgboost_traffic_model.pkl"
    )


@st.cache_data
def load_data():

    data = np.load(
        "data/processed_features.npz"
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    data.close()

    return X_test, y_test


model = load_model()
X_test, y_test = load_data()

# -------------------------------
# Congestion thresholds
# -------------------------------

LOW_THRESHOLD = 0.19710415693601122
HIGH_THRESHOLD = 0.4040168145726296


def classify_congestion(value):

    if value < LOW_THRESHOLD:
        return "LOW"

    elif value < HIGH_THRESHOLD:
        return "MEDIUM"

    return "HIGH"


# -------------------------------
# Header
# -------------------------------

st.title("🚦 TrafficSense")

st.subheader(
    "AI-Powered Traffic Flow Prediction & Congestion Analysis"
)

st.markdown(
    "Predict traffic conditions using machine learning "
    "and classify congestion levels."
)


# -------------------------------
# Make predictions
# -------------------------------

predictions = model.predict(X_test)


# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Prediction")

sample_number = st.sidebar.slider(
    "Select a test sample",
    min_value=1,
    max_value=len(X_test),
    value=1
)

index = sample_number - 1

predicted_value = predictions[index]

actual_value = y_test[index]

congestion = classify_congestion(
    predicted_value
)


# -------------------------------
# Metrics
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Predicted Traffic",
        f"{predicted_value:.3f}"
    )

with col2:

    st.metric(
        "Actual Traffic",
        f"{actual_value:.3f}"
    )

with col3:

    st.metric(
        "Congestion Level",
        congestion
    )


# -------------------------------
# Congestion message
# -------------------------------

if congestion == "LOW":

    st.success(
        "🟢 Low traffic — traffic flow is relatively smooth."
    )

elif congestion == "MEDIUM":

    st.warning(
        "🟡 Medium traffic — moderate congestion expected."
    )

else:

    st.error(
        "🔴 High traffic — heavy congestion expected."
    )


# -------------------------------
# Prediction comparison
# -------------------------------

st.header("📊 Prediction vs Actual")

comparison = pd.DataFrame(
    {
        "Actual Traffic": y_test[:100],
        "Predicted Traffic": predictions[:100]
    }
)

st.line_chart(comparison)


# -------------------------------
# Congestion distribution
# -------------------------------

st.header("🚦 Congestion Distribution")

low_count = np.sum(
    predictions < LOW_THRESHOLD
)

medium_count = np.sum(
    (predictions >= LOW_THRESHOLD)
    & (predictions < HIGH_THRESHOLD)
)

high_count = np.sum(
    predictions >= HIGH_THRESHOLD
)


distribution = pd.DataFrame(
    {
        "Congestion": [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        "Count": [
            low_count,
            medium_count,
            high_count
        ]
    }
)

st.bar_chart(
    distribution.set_index("Congestion")
)


# -------------------------------
# Model information
# -------------------------------

st.header("🤖 Model Information")

st.write(
    """
    **Model:** XGBoost Regressor

    **Purpose:** Traffic flow prediction

    **Features:** Last Traffic Value, Average Traffic,
    Maximum Traffic, Minimum Traffic, Traffic Variation

    **R² Score:** 0.8437

    **RMSE:** 0.0809
    """
)