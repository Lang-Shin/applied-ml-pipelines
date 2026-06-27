"""
    E-Commerce Purchase Prediction
    Feat : Time_on_Page, Pages_Visited, Device_Mobile, Discount_Used, Items_in_Cart, Past_Purchases, Premium_User
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


df = pd.read_csv("LogisticRegression/Practice1/ecommerce_analytics.csv")

X = df.drop("Premium_User", axis=1)
y = df["Premium_User"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y, 
    test_size = 0.2,
    random_state = 42
)


model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

pred_proba = model.predict_proba(X_test)[:,1]
loss = log_loss(y_test,pred_proba)

for predict, actual in zip(predictions, y_test):
    print(f"Prediction : {predict}  --  Actual : {actual}")
    
print("\n\nModel's Accuracy : ", int(model.score(X_test, y_test) * 100))
print("\nBias : ", model.intercept_[0], "\n")

for i in range(len(model.coef_)):
    for j in range(len(X.columns)):
        print(f"Feature : {X.columns[j]}")
        print(f"Weight  : {model.coef_[i][j]}\n")

print("Loss : ", loss)