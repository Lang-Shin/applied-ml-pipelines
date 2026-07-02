import numpy as np
import pandas as pd


np.random.seed(42)
n = 3000


age = np.random.randint(23, 65, size=n)
income = np.random.randint(21914, 131484, size=n)
yr_at_com = np.random.randint(1, 42, size=n)
distance_home = np.random.randint(1, 25, size=n)
overtime = np.random.randint(1, 60, size=n)
job_satis = np.random.choice([1,2,3,4,5], size=n)
performance = np.random.choice([1,2,3,4,5], size=n)
department = np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], size=n)
educ = np.random.choice(['Bachelor', 'Masters', 'PhD'], size=n)
balance = np.random.choice([1,2,3,4,5], size=n)

score = (
    -4.5 +  
    (overtime * 0.05) +
    (6 - job_satis) * 0.4 +
    (6 - balance) * 0.3 +
    (distance_home * 0.05) +
    (131484 - income) / 100000 + 
    (42 - yr_at_com) * 0.02 +
    (6 - performance) * 0.1
)

sigmoid = 1 / (1 + np.exp(-score))

attrition = np.where(sigmoid > np.random.uniform(0, 1, size=n), 1, 0)

df = pd.DataFrame({
    "Age" : age,
    "Income" : income,
    "YratCom" : yr_at_com,
    "DistFromHome" : distance_home,
    "Overtime" : distance_home,
    "JobSatist" : job_satis,
    "Performance" : performance,
    "Department" : department,
    "Educ" : educ,
    "WorkLifeBalance" : balance,
    "Attrition" : attrition
})

df.to_csv("KNN/Practice3/employee_att.csv", index=False)

print(f"\nDataset generated successfully!\nRow : {df.shape[0]}\tColumn : {df.shape[1]}")