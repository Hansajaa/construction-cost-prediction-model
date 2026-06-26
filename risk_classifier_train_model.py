import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("construction_dataset_for_classifier.csv")

# Explore
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert Material_Cost_USD -> LKR
USD_TO_LKR = 300  # update to current rate as needed
df['Material_Cost_LKR'] = df['Material_Cost_USD'] * USD_TO_LKR
df = df.drop(columns=['Material_Cost_USD'])

# Drop Task_ID - it's just an identifier, not a predictive feature
df = df.drop(columns=['Task_ID', 'Resource_Constraint_Score', 'Site_Constraint_Score'])

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define feature columns (no categorical INPUT features this time -
# Risk_Level is the TARGET, not a feature)
numerical_features = ['Task_Duration_Days', 'Labor_Required', 'Equipment_Units',
                       'Material_Cost_LKR', 'Start_Constraint',
                       'Dependency_Count']

# Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features)
])

from sklearn.model_selection import train_test_split

X = df.drop('Risk_Level', axis=1)   # Features
y = df['Risk_Level']                 # Target (Low / Medium / High)

# stratify=y keeps the same Low/Medium/High proportions in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ====== CHANGED: Regressor -> Classifier ======
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ====== CHANGED: MAE/R2 -> Accuracy/F1 ======
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    results[name] = {"Accuracy": acc, "F1": f1, "pipeline": pipeline}
    print(f"{name:25s} -> Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

from sklearn.model_selection import GridSearchCV

# Tune Gradient Boosting (adjust based on which model wins above)
param_grid = {
    'model__n_estimators':      [100, 200],
    'model__max_depth':         [3, 5, 7],
    'model__learning_rate':     [0.05, 0.1, 0.2],
    'model__min_samples_split': [2, 5]
}

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(random_state=42))
])

# ====== CHANGED: scoring metric for classification ======
grid_search = GridSearchCV(best_pipeline, param_grid, cv=5,
                           scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
final_model = grid_search.best_estimator_

# Evaluate
y_pred_final = final_model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_final, average='weighted'):.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred_final))

# ====== CHANGED: Confusion Matrix instead of scatter plot ======
cm = confusion_matrix(y_test, y_pred_final, labels=final_model.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=final_model.classes_, yticklabels=final_model.classes_)
plt.xlabel("Predicted Risk Level")
plt.ylabel("Actual Risk Level")
plt.title("Risk Level - Confusion Matrix")
plt.tight_layout()
plt.savefig("risk_prediction_results.png")
print("Chart saved!")

import joblib
joblib.dump(final_model, "risk_classifier_model.pkl")
print("Model saved successfully!")