import numpy as np
import pandas as pd


np.random.seed(42)
n = 5000


employee_id = np.arange(1001, 1001+n)
age = np.random.randint(23, 64, size=n)
years_of_exp = np.random.randint(0, 21, size=n)
per_score = np.random.randint(50, 101, size=n)
train_hrs = np.random.randint(0, 61, size=n)
project_complete = np.random.randint(1,16, size=n)
salary = np.random.randint(25000, 120001, size=n)

score = (
    per_score * 0.45
    + years_of_exp * 1.6
    + train_hrs * 0.35
    + project_complete * 1.8
)

noise = np.random.normal(0, 8, size=n)

score += noise

promote = (score > 70).astype(int)

df = pd.DataFrame({
    "Emp_ID" : employee_id,
    "Age" : age,
    "Yrs_Exp" : years_of_exp,
    "Performance_Score" : per_score,
    "Train_Hrs" : train_hrs,
    "Proj_Complete" : project_complete,
    "Salary" : salary,
    "Promoted" : promote
})

df.to_csv("KNN/Practice1/employee_description.csv", index=False)

print(f"\nData generated successfully\nRows : {df.shape[0]}\tColumn : {df.shape[1]}")