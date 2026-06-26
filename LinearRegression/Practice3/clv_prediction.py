"""
E-Commerce Customer Lifetime Value(CLV) Prediction based on feature:
Feat : TenureMonths, LoginFrequency, CartAbandonmentRate, PagesViewed, SupportTickets, IsPremium, CampaignAudience, TotalSpend
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def fixMarks(df, cols):
    """
        Fix col from str -> float
        removes str val turns into NaN, NaN -> median
    """
    
    df[cols] = pd.to_numeric(df[cols], errors='coerce')
    
    median = df[cols].median()
    
    df[cols] = df[cols].fillna(median)
    
    return df



def encode_categorical_feat(df, cols):
    """
        Convert Yes to 1 and No to 0
        Convert other str to 0
    """
    
    df[cols] = df[cols].replace({
        "Yes" : 1,
        "No" : 0,
        "MISSING" : 0
    })
    
    return df
    

df = pd.read_csv("LinearRegression/Practice3/customer_spend.csv")
df = df[df['TotalSpend'] < 30000]
df = fixMarks(df, 'TenureMonths')
df = fixMarks(df, 'SupportTickets')
df = encode_categorical_feat(df, ['IsPremium', 'CampaignAudience'])

X = df.drop('TotalSpend', axis=1)
y = df['TotalSpend']

stan_scaler = StandardScaler()
X_scale = stan_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scale,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

for predict, actual in zip(predictions, y_test):
    print(f"Prediction : {predict:.2f}\t--\tActual : {actual}")
    

rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n\nRoot Mean Squared Error : ", rmse)
print("R^2 Score : ", r2)


#? VISUALIZATION's
#* Model's Performance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, predictions, color="#076100")
axes[0].set_xlabel("Actual Spend")
axes[0].set_ylabel("Predictions")
axes[0].grid(True, alpha=0.6)
axes[0].set_title("Model's Performance")

#! Model's Error
residuals = y_test - predictions
axes[1].scatter(predictions, residuals, color="#7F0101")
axes[1].set_xlabel("Predictions")
axes[1].set_ylabel("Error's")
axes[1].axhline(y=0, color="#330000")
axes[1].set_title("Model's Error")
axes[1].grid(True, alpha=0.6)

plt.tight_layout()
plt.show()
