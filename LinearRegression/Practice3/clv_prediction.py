"""
E-Commerce Customer Lifetime Value(CLV) Prediction based on feature:
Feat : TenureMonths, LoginFrequency, CartAbandonmentRate, PagesViewed, SupportTickets, IsPremium, CampaignAudience, TotalSpend
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""

!   TenureMonths -> str(?)
!   SupportTickets -> str(?)
!   IsPremium -> str(y/n)
!   CampaignAudience -> str(y/n/MISSING)   

"""

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
    df[cols] = df[cols].replace({
        "Yes" : 1,
        "No" : 0,
        "MISSING" : 0
    })
    
    return df
    

df = pd.read_csv("LinearRegression/Practice3/customer_spend.csv")
df = fixMarks(df, 'TenureMonths')   #? Fixed TenureMonths
df = fixMarks(df, 'SupportTickets') #? Fixed SupportTickets
df = encode_categorical_feat(df, ['IsPremium', 'CampaignAudience'])

print(df.info())
print("\n\n", df)