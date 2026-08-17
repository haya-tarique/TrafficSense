import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load processed data
data = np.load("data/processed_features.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]


print("X train:", X_train.shape)
print("y train:", y_train.shape)
print("X test :", X_test.shape)
print("y test :", y_test.shape)


# Create XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# Train
print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed!")


# Predict
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nXGBoost Performance:")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# Save model
joblib.dump(
    model,
    "models/xgboost_traffic_model.pkl"
)

print("\nXGBoost model saved successfully!")