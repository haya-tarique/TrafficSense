import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load ONLY the processed features
processed_data = np.load("data/processed_features.npz")

X_train = processed_data["X_train"]
y_train = processed_data["y_train"]

X_test = processed_data["X_test"]
y_test = processed_data["y_test"]


# Check the data
print("X train:", X_train.shape)
print("y train:", y_train.shape)

print("X test :", X_test.shape)
print("y test :", y_test.shape)


# Create Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Train
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# Predict
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# Save model
joblib.dump(
    model,
    "models/traffic_model.pkl"
)

print("\nModel saved successfully!")