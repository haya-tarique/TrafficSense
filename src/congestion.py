import numpy as np
import joblib


# --------------------------------
# Load processed data
# --------------------------------

data = np.load("data/processed_features.npz")

X_test = data["X_test"]
y_test = data["y_test"]


# --------------------------------
# Load trained XGBoost model
# --------------------------------

model = joblib.load(
    "models/xgboost_traffic_model.pkl"
)


# --------------------------------
# Congestion thresholds
# --------------------------------

low_threshold = 0.19710415693601122
high_threshold = 0.4040168145726296


# --------------------------------
# Congestion classification
# --------------------------------

def classify_congestion(value):

    if value < low_threshold:
        return "LOW"

    elif value < high_threshold:
        return "MEDIUM"

    else:
        return "HIGH"


# --------------------------------
# Make predictions
# --------------------------------

predictions = model.predict(X_test)


# --------------------------------
# Display predictions
# --------------------------------

print("\nTraffic Predictions")
print("-------------------")

for i in range(10):

    predicted = predictions[i]

    actual = y_test[i]

    congestion = classify_congestion(predicted)

    print(
        f"Sample {i + 1}: "
        f"Predicted = {predicted:.4f}, "
        f"Actual = {actual:.4f}, "
        f"Congestion = {congestion}"
    )


# --------------------------------
# Count congestion levels
# --------------------------------

low_count = 0
medium_count = 0
high_count = 0


for prediction in predictions:

    level = classify_congestion(prediction)

    if level == "LOW":
        low_count += 1

    elif level == "MEDIUM":
        medium_count += 1

    else:
        high_count += 1


# --------------------------------
# Display summary
# --------------------------------

print("\nCongestion Summary")
print("------------------")

print("LOW    :", low_count)
print("MEDIUM :", medium_count)
print("HIGH   :", high_count)