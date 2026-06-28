"""
    Heart Disease Risk Prediction feat : LogisticRegression
    Data Feat : Age, Resting_BP, Cholesterol, Max_Heart_Rate, BMI, Exercise_Angina, Target
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def replace_zero_with_median(df, col, value):
    df[col] = df[col].replace(0, value)


df = pd.read_csv("LogisticRegression/Practice2/cardio_risk_dataset.csv")

X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# Get the column median of columns with zeroes values
chol_med = X_train['Cholesterol'].median()
bmi_med = X_train['BMI'].median()

# replace invalid zero values with column median on training dataset
replace_zero_with_median(X_train, 'Cholesterol', chol_med)
replace_zero_with_median(X_train, 'BMI', bmi_med)

# replace invalid zero values with column median on testing dataset
replace_zero_with_median(X_test, 'Cholesterol', chol_med)
replace_zero_with_median(X_test, 'BMI', bmi_med)

need_scale_feat = ["Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate", "BMI"]

preprocessor = ColumnTransformer(
    transformers = [("num", StandardScaler(), need_scale_feat)],
    remainder = "passthrough"
)

X_train_scale = preprocessor.fit_transform(X_train)
X_test_scale = preprocessor.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scale, y_train)

predictions = model.predict(X_test_scale)

print(f"\nBias : ", model.intercept_[0])
for feature, weight in zip(X.columns, model.coef_[0]):
    print(f"{feature:18} -- {weight:.4f}")

pred_proba = model.predict_proba(X_test_scale)[:,1]
loss = log_loss(y_test, pred_proba)
print(f"\nLoss : {loss}\n")

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy : {(accuracy*100):.2f}%\n")

for pred, act in zip(predictions, y_test):
    print(f"Prediction : {pred}  -- Actual : {act}")
    
    
# VISUALIZATION's
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[0,0])
axes[0,0].grid(True, alpha=0.6)
axes[0,0].set_title("Confusion Matrix")

RocCurveDisplay.from_estimator(model, X_test_scale, y_test, ax=axes[0,1])
axes[0,1].grid(True, alpha=0.6)
axes[0,1].set_title("Roc Curve")

axes[1,0].bar(X.columns, model.coef_[0], color="#14581C")
axes[1,0].set_xlabel("Features")
axes[1,0].set_ylabel("Weights")
axes[1,0].set_title("Feature Coefficient")
axes[1,0].grid(True, alpha=0.6)

axes[1,1].scatter(range(len(y_test)), y_test, label="Actual", color="#215818")
axes[1,1].scatter(range(len(predictions)), predictions, marker='x', label="Predictions",color="#930F0F")
axes[1,1].grid(True, alpha=0.6)
axes[1,1].set_title("Actual vs Prediction")
axes[1,1].legend()

plt.tight_layout()
plt.show()