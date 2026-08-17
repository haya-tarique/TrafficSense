from scipy.io import loadmat
import numpy as np

# Load dataset
data = loadmat("data/traffic_dataset.mat")

# Get training and testing data
X_train_raw = data["tra_X_tr"]
X_test_raw = data["tra_X_te"]

Y_train = data["tra_Y_tr"]
Y_test = data["tra_Y_te"]

print("Original shapes:")
print("X train:", X_train_raw.shape)
print("X test :", X_test_raw.shape)
print("Y train:", Y_train.shape)
print("Y test :", Y_test.shape)

# Convert sparse matrices into normal NumPy arrays
X_train = np.stack([
    X_train_raw[0, i].toarray()
    for i in range(X_train_raw.shape[1])
])

X_test = np.stack([
    X_test_raw[0, i].toarray()
    for i in range(X_test_raw.shape[1])
])

# Transpose Y so that:
# rows = samples
# columns = sensors
Y_train = Y_train.T
Y_test = Y_test.T

print("\nAfter processing:")

print("X train:", X_train.shape)
print("X test :", X_test.shape)

print("Y train:", Y_train.shape)
print("Y test :", Y_test.shape)

print("\nOne training sample shape:")
print(X_train[0].shape)

print("\nOne target sample shape:")
print(Y_train[0].shape)