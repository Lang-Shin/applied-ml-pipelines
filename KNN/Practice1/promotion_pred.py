import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
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

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
df_result = cv_results[['param_classifier__n_neighbors', 'mean_test_score', 'std_test_score']]
df_result = df_result.sort_values(by='param_classifier__n_neighbors').reset_index(drop=True)

predictions = grid_search.predict(X_test)

best_k = df_result.iloc[df_result['mean_test_score'].idxmax()]
worst_k = df_result.iloc[df_result['mean_test_score'].idxmin()]

print("\n\n",df_result)
print(f"\nBest K :\n{best_k}")
print(f"\nWorst K:\n{worst_k}")
print("\n\n", classification_report(y_test, predictions))

for pred, act in zip(predictions[:20], y_test.iloc[:20]):
    print(f"Prediction : {pred}  -- Actual : {act}")
    


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[0])
axes[0].set_title("Confusion Matrix")

bar = axes[1].bar(df_result['param_classifier__n_neighbors'], df_result['mean_test_score'], color="#136D1F", edgecolor="#000000", linewidth=1.2)
axes[1].set_xlabel("K")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("K vs Accuracy")
axes[1].set_xticks(df_result['param_classifier__n_neighbors'])
axes[1].grid(True, alpha=0.6)
axes[1].bar_label(bar, padding=3, fmt='%.4f')
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.show()