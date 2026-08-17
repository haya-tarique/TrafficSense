import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load processed data
data = np.load("data/processed_features.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]


# Check shapes
print("X train:", X_train.shape)
print("y train:", y_train.shape)
print("X test :", X_test.shape)
print("y test :", y_test.shape)


# Create Linear Regression model
model = LinearRegression()


# Train
print("\nTraining Linear Regression...")

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


# Display results
print("\nLinear Regression Performance:")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)