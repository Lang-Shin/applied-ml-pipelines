"""
    Premium Subscription Prediction
    Feat : Age, Monthly_Hours_Log, Customer_Support_Calls, Device_Type, Discount_Offered, Satisfaction_Score, Premium_User
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df = pd.read_csv("LogisticRegression/Practice3/user_desc.csv")

X = df.drop('Premium_User', axis=1)
y = df['Premium_User']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 42,
    test_size = 0.2
)

continous_feat = ['Age', 'Monthly_Hours_Log', 'Customer_Support_Calls', 'Discount_Offered', "Satisfaction_Score"]

preprocessor = ColumnTransformer(
    transformers = [
        ("feat_scale", StandardScaler(), continous_feat),
        ("converted_categorial", OneHotEncoder(), ['Device_Type'])
    ],
    remainder = "passthrough"
)


model = Pipeline([
    ("transformers", preprocessor),
    ("classifier", LogisticRegression)
])
