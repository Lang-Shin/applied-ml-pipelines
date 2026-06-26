"""
Predicting House prices based on feat:
SquareFootage, Bedrooms, Bathrooms, Age, GarageSize, HasPool, IsNearSchool, SellingPrice
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def treatNaNs(df, col):
    
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    median = df[col].median()
    
    df[col] = df[col].fillna(median) 
    
    return df   
    

#* Categorial Data -> Numerical Data
def encode_categorial_feat(df, col):
    
    df[col] = df[col].replace({
        "Yes" : 1,
        "No" : 0
    })
    
    return df

df = pd.read_csv("LinearRegression/Practice2/housing_data_medium.csv")
df = encode_categorial_feat(df, ['HasPool', 'IsNearSchool'])
df = treatNaNs(df, 'SquareFootage')
df = treatNaNs(df, 'Bedrooms')

X = df.drop("SellingPrice", axis=1)
y = df['SellingPrice']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

for predict, actual in zip(predictions, y_test):
    print(f"Prediction : {predict:.2f}K\t--\tActual Price : {actual}")

print(f"\nRoot Mean Squared Error : {rmse}")
print(f"R^2 : {r2}")


fig, axes = plt.subplots(1, 2, figsize=(12,5))

#* Model's Performance
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
axes[0].plot([min_val, max_val], [min_val, max_val], color="#024215")
axes[0].scatter(y_test, predictions, color="#029C1C")
axes[0].grid(True, alpha=0.6)
axes[0].set_ylabel("Predictions")
axes[0].set_xlabel("Actual Value")
axes[0].set_title("Model's Performance")

#! Model's Error
residuals = y_test - predictions
axes[1].scatter(predictions, residuals, color="#BB0101")
axes[1].axhline(y=0, color="#000000")
axes[1].grid(True, alpha=0.6)
axes[1].set_xlabel("Performance")
axes[1].set_ylabel("Error's")
axes[1].set_title("Model's Error")

plt.tight_layout()
plt.show()