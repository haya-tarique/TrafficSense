from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------

data = loadmat("data/traffic_dataset.mat")

X_train_raw = data["tra_X_tr"]
Y_train = data["tra_Y_tr"]

# Convert X into normal NumPy arrays
X_train = np.stack([
    X_train_raw[0, i].toarray()
    for i in range(X_train_raw.shape[1])
])

# Transpose Y
Y_train = Y_train.T

print("X shape:", X_train.shape)
print("Y shape:", Y_train.shape)


# -----------------------------
# 2. Basic statistics
# -----------------------------

print("\nTraffic Statistics")

print("Minimum traffic value:", Y_train.min())
print("Maximum traffic value:", Y_train.max())
print("Average traffic value:", Y_train.mean())


# -----------------------------
# 3. Traffic distribution
# -----------------------------

plt.figure(figsize=(10, 5))

plt.hist(Y_train.flatten(), bins=50)

plt.title("Distribution of Traffic Values")
plt.xlabel("Traffic Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.show()


# -----------------------------
# 4. Traffic pattern of Sensor 1
# -----------------------------

sensor_1 = Y_train[:, 0]

plt.figure(figsize=(12, 5))

plt.plot(sensor_1)

plt.title("Traffic Pattern - Sensor 1")
plt.xlabel("Time Step")
plt.ylabel("Traffic Value")

plt.tight_layout()

plt.show()


# -----------------------------
# 5. Compare multiple sensors
# -----------------------------

plt.figure(figsize=(12, 5))

for sensor in range(5):
    plt.plot(
        Y_train[:, sensor],
        label=f"Sensor {sensor + 1}"
    )

plt.title("Traffic Patterns Across Sensors")
plt.xlabel("Time Step")
plt.ylabel("Traffic Value")

plt.legend()

plt.tight_layout()

plt.show()