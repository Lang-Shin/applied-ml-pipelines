"""
    Premium Subscription Prediction
    Feat : Age, Monthly_Hours_Log, Customer_Support_Calls, Device_Type, Discount_Offered, Satisfaction_Score, Premium_User
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

proba = model.predict_proba(X_test)[:,1]
loss = log_loss(y_test, proba)

classifier = model.named_steps['classifier']

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print(f"\n{cm}")
print(f"\nAccuracy : {(accuracy*100):.2f}%")
print(f"\nLoss : {loss:.2f}")
print(f"\nBias : {classifier.intercept_[0]}")

feat_names = model.named_steps['transformers'].get_feature_names_out()

for feat, weight in zip(feat_names, classifier.coef_[0]):
    print(f"{feat:50} -- {weight}")


fig, axes = plt.subplots(1,3, figsize=(12,5))
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[0])
axes[0].set_title("Confusion Matrix")

axes[1].bar(feat_names, classifier.coef_[0], color="#000000")
axes[1].grid(True, alpha=0.6)
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Weights")
axes[1].set_title("Feature Coefficient")
axes[1].tick_params(axis='x', rotation=60)

RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[2])
axes[2].grid(True, alpha=0.6)
axes[2].set_title("Roc Curve")

plt.tight_layout()
plt.show()