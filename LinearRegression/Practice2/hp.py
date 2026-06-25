"""
Predicting House prices based on feat:
SquareFootage, Bedrooms, Bathrooms, Age, GarageSize, HasPool, IsNearSchool, SellingPrice
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def encode_categorial_feat(df, col):
    
    df[col] = df[col].replace({
        "Yes" : 1,
        "No" : 0
    })
    
    return df

df = pd.read_csv("LinearRegression/Practice2/housing_data_medium.csv")
df = encode_categorial_feat(df, ['HasPool', 'IsNearSchool'])



print(df)