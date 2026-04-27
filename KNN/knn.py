import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def custom_train_test(features, labels, train_size=0.8):
    n = len(features)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_point = int(n * train_size)
    train_data_id = indices[:split_point]
    test_data_id = indices[split_point:]
    return features[train_data_id], features[test_data_id], labels[train_data_id], labels[test_data_id]

def predict_customer(customer, x_train, y_train, k):
    diff = customer - x_train
    distances = np.sqrt(np.sum(diff**2, axis=1))
    nearest_neighbor_indices = np.argsort(distances)[:k]
    neighbor_labels = y_train[nearest_neighbor_indices]
    return 1 if np.sum(neighbor_labels) > (k/2) else 0

# Load and prepare data
df = pd.read_csv("customer_segmentation_data.csv")
data_arr = df.values
features = data_arr[:, :-1]
labels = data_arr[:, -1]

# Perform split
x_train, x_test, y_train, y_test = custom_train_test(features, labels)

# Choose a test customer
customer = x_test[0]
customer_label = y_test[0]

# Perform prediction
prediction = predict_customer(customer, x_train, y_train, 5)

print(f"Prediction: {prediction}")
print(f"Actual Label: {customer_label}")

#Visualization
plt.figure(figsize=(10, 6))

# Plot training data by label
for label in np.unique(y_train):
    mask = (y_train == label)
    plt.scatter(x_train[mask, 0], x_train[mask, 1], label=f'Train Class {int(label)}', alpha=0.6)

# Plot the test customer as a large red star
plt.scatter(customer[0], customer[1], c='red', marker='*', s=200, label='Test Customer')

plt.title('K-NN Classification Result')
plt.xlabel('Annual_Income_k')
plt.ylabel('Spending_Score')
plt.legend()
plt.grid(True)
plt.show()