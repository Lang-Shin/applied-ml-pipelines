import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv("KNN/Practice1/employee_description.csv")

X = df.drop(['Emp_ID','Promoted'], axis=1)
y = df['Promoted']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocess = ColumnTransformer(
    transformers = [('preprocessing', StandardScaler(), X_train.columns)]
)


pipeline = Pipeline([
    ('preprocessor', preprocess),
    ('classifier', KNeighborsClassifier())
])

param_grid = {
    'classifier__n_neighbors' : [3, 5, 7, 9]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

print("\n\n", classification_report(y_test, predictions), "\n\n")

for pred, act in zip(predictions[:20], y_test[:20]):
    print(f"Prediction : {pred}  -- Actual : {act}")
    

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()