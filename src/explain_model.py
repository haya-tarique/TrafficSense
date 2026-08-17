import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# --------------------------------
# Load processed data
# --------------------------------

data = np.load("data/processed_features.npz")

X_test = data["X_test"]


# --------------------------------
# Load trained XGBoost model
# --------------------------------

model = joblib.load(
    "models/xgboost_traffic_model.pkl"
)


# --------------------------------
# Feature names
# --------------------------------

feature_names = [
    "Last Traffic Value",
    "Average Traffic",
    "Maximum Traffic",
    "Minimum Traffic",
    "Traffic Variation"
]


# --------------------------------
# Select a small sample
# --------------------------------

X_sample = X_test[:500]


# --------------------------------
# Create SHAP explainer
# --------------------------------

print("Creating SHAP explanations...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)


# --------------------------------
# Global feature importance
# --------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "images/shap_feature_importance.png"
)

plt.show()


print("SHAP analysis completed!")

print(
    "Saved to images/shap_feature_importance.png"
)