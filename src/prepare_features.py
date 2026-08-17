from scipy.io import loadmat
import numpy as np


# ==========================================
# 1. Load original dataset
# ==========================================

data = loadmat("data/traffic_dataset.mat")

X_train_raw = data["tra_X_tr"]
X_test_raw = data["tra_X_te"]

Y_train_raw = data["tra_Y_tr"]
Y_test_raw = data["tra_Y_te"]


# ==========================================
# 2. Convert sparse matrices
# ==========================================

X_train_original = np.stack([
    X_train_raw[0, i].toarray()
    for i in range(X_train_raw.shape[1])
])

X_test_original = np.stack([
    X_test_raw[0, i].toarray()
    for i in range(X_test_raw.shape[1])
])


# Transpose targets
Y_train_original = Y_train_raw.T
Y_test_original = Y_test_raw.T


print("Original data:")
print("X train:", X_train_original.shape)
print("X test :", X_test_original.shape)
print("Y train:", Y_train_original.shape)
print("Y test :", Y_test_original.shape)


# ==========================================
# 3. Feature Engineering
# ==========================================

def create_features(X, Y):

    feature_rows = []
    target_values = []

    for sample_index in range(X.shape[0]):

        sample = X[sample_index]

        # Five features for every sensor

        last_value = sample[:, -1]

        mean_value = sample.mean(axis=1)

        max_value = sample.max(axis=1)

        min_value = sample.min(axis=1)

        std_value = sample.std(axis=1)


        # Create one row for every sensor

        for sensor in range(36):

            row = [
                last_value[sensor],
                mean_value[sensor],
                max_value[sensor],
                min_value[sensor],
                std_value[sensor]
            ]

            feature_rows.append(row)

            target_values.append(
                Y[sample_index, sensor]
            )


    return np.array(feature_rows), np.array(target_values)


# ==========================================
# 4. Create processed datasets
# ==========================================

X_train_features, y_train = create_features(
    X_train_original,
    Y_train_original
)

X_test_features, y_test = create_features(
    X_test_original,
    Y_test_original
)


# ==========================================
# 5. Check processed data
# ==========================================

print("\nAfter feature engineering:")

print("X train:", X_train_features.shape)
print("y train:", y_train.shape)

print("X test :", X_test_features.shape)
print("y test :", y_test.shape)


# ==========================================
# 6. Show example
# ==========================================

print("\nFirst training row:")
print(X_train_features[0])

print("\nFirst target:")
print(y_train[0])


# ==========================================
# 7. Save processed data
# ==========================================

np.savez(
    "data/processed_features.npz",
    X_train=X_train_features,
    y_train=y_train,
    X_test=X_test_features,
    y_test=y_test
)


print("\nProcessed features saved successfully!")