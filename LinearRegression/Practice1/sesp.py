"""
Student Exam Score Prediction
feat : StudyHours, Attendance, AssignmentsCompleted, QuizAverage, FinalExamScore
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv("LinearRegression/Practice1/student_performance.csv")

X = df.drop("FinalExamScore", axis=1)
y = df['FinalExamScore']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

for pred, actual in zip(predictions, y_test):
    print(f"Prediction : {pred:.2f}\t-->\tActual Score : {actual}")


fig, axes = plt.subplots(1, 2, figsize=(10,5))

# Model's Performance
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

axes[0].plot([min_val, max_val], [min_val, max_val], color="#0F317F")
axes[0].scatter(y_test, predictions, color="#197E1C")
axes[0].grid(True, alpha=0.6)
axes[0].set_ylabel("Model's Predictions")
axes[0].set_xlabel("Actual Scores")
axes[0].set_title("Model's Performance")

# Model's Error
residuals = y_test - predictions
axes[1].scatter(predictions, residuals, color="#812525")
axes[1].axhline(y=0, color="#000000")
axes[1].grid(True, alpha=0.6)
axes[1].set_ylabel("Model's Error")
axes[1].set_xlabel("Predicted Values")
axes[1].set_title("Model's Error")

plt.tight_layout()
plt.show()