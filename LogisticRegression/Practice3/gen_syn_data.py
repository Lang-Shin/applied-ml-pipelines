import numpy as np
import pandas as pd


np.random.seed(42)
n = 5000

# Generate Synthetic Features
age = np.random.randint(18, 65, size=n)
monthly_hrs_log = np.random.normal(25, 8, size=n)
cs_calls = np.random.poisson(1.5, size=n)
device_type = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n)
discount_offered = np.random.choice([0,1], p=[0.7, 0.3], size=n)
satisfaction_score = np.random.randint(1, 6, size=n)

log_odds = (
    0.04 * monthly_hrs_log
    - 0.03 * cs_calls
    + 0.5 * discount_offered
    + 0.2 * satisfaction_score
    - 1.5
)

prob = 1 / (1 + np.exp(-log_odds))

premium = np.random.binomial(1, prob)

df = pd.DataFrame({
    "Age" : age,
    "Monthly_Hours_Log" : monthly_hrs_log,
    "Customer_Support_Calls" : cs_calls,
    "Device_Type" : device_type,
    "Discount_Offered" : discount_offered,
    "Satisfaction_Score" : satisfaction_score,
    "Premium_User" : premium
})

df.to_csv("LogisticRegression/Practice3/user_desc.csv")

print(f"\nData generated successfully\nRows : {df.shape[0]}\tColumns : {df.shape[1]}")