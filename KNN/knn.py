import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def custom_train_test(features, labels, train_size=0.8):
    n = len(features)

    indices = np.arange(n)

    np.random.shuffle(indices)

    split_point = (indices*train_size)

    train_data_id = indices[:int(split_point)]
    test_data_id = indices[int(split_point):]

    x_train, x_test = features[train_data_id], features[test_data_id]
    y_train, y_test = labels[train_data_id], labels[test_data_id]

    return x_train, x_test, y_train, y_test

def predict_customer(customer, x_train, y_train, k):
    diff = customer - x_train

    distances = np.sqrt(np.sum(diff**2, axis=1))

    nearest_neighbor_indices = np.argsort(distances)[:k]

    neighbor_labels = y_train[nearest_neighbor_indices]

    return 1 if np.sum(neighbor_labels) > (k/2) else 0


df = pd.read_csv("customer_segmentation_data.csv")

data_arr = df.values

features = data_arr[:, :-1]
labels = data_arr[:,-1]

print(df)

print(features)
print(labels)