"""
    Generate Synthetic Data for LogisticRegression Practice2
    Feat : age, restingbp, cholesterol, max_heart_rate, bmi, exercise_angina, target
"""


import numpy as np
import pandas as pd


np.random.seed(42)
n = 3000

# Generate Synthetic Feature
age = np.random.randint(29, 75, size=n)
resting_bp = np.random.randint(95, 180, size=n)
cholesterol = np.random.randint(140, 360, size=n)
max_hr = np.random.randint(90, 202, size=n)
bmi = np.round(np.random.uniform(18.5, 39.9, size=n), 1)
exercise_angina = np.random.choice([0,1], p=[0.6, 0.4], size=n)

# Generate som realistic "0" vales to simulate missing data for Preprocessing
for i in np.random.choice(n, 15, replace=False):
    cholesterol[i] = 0
    
for i in np.random.choice(n, 15, replace=False):
    bmi[i] = 0.0
    
log_odds = (
    0.04 * age
    + 0.015 * resting_bp
    + 0.008 * cholesterol
    - 0.03 * max_hr
    + 0.12 * bmi
    + 1.5 * exercise_angina
    - 7.5
)

prob = 1 / (1 + np.exp(-log_odds))
target = np.where(prob > np.uniform(0, 1, size=n), 1, 0)

# Create DataFrame
df = pd.DataFrame({
    "Age" : age,
    "Resting_BP" : resting_bp,
    "Cholesterol" : cholesterol,
    "Max_Heart_Rate" : max_hr,
    "BMI" : bmi,
    "Exercise_Angina" : exercise_angina,
    "Target" : target
})

# Save to CSV
df.to_csv("LogisticRegression/Practice2/cardio_risk_dataset.csv")

print(f"\n\nDataset generated successfully\nRows : {df.shape[0]}\tColumns : {df.shape[1]}")