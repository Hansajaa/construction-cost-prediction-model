import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATASET_CANDIDATES = [
    Path('construction_data_balanced_v2.csv'),
    Path('construction_data_balanced.csv'),
    Path('construction_data_augmented.csv'),
    Path('construction_data.csv'),
]

dataset_path = next((candidate for candidate in DATASET_CANDIDATES if candidate.exists()), None)
if dataset_path is None:
    raise FileNotFoundError('No construction dataset file was found for training.')

print(f'Using dataset: {dataset_path.name}')
df = pd.read_csv(dataset_path)

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

df['total_area_sqft'] = df['total_area_sqft'].fillna(df['total_area_sqft'].median())
df['project_type'] = df['project_type'].fillna(df['project_type'].mode()[0])

Q1 = df['total_cost'].quantile(0.25)
Q3 = df['total_cost'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['total_cost'] >= Q1 - 1.5 * IQR) & (df['total_cost'] <= Q3 + 1.5 * IQR)]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ['project_type', 'location_zone', 'material_grade', 'soil_type']
numerical_features = [
    'total_area_sqft',
    'num_floors',
    'labor_cost_per_day',
    'project_duration_days',
    'num_workers',
]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

from sklearn.model_selection import GridSearchCV, train_test_split

X = df.drop('total_cost', axis=1)
y = df['total_cost']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples:  {X_test.shape[0]}')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name:25s} -> MAE: LKR {mae:,.2f} | R2: {r2:.4f}')

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__min_samples_split': [2, 5],
}

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42)),
])

grid_search = GridSearchCV(
    best_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
final_model = grid_search.best_estimator_

y_pred_final = final_model.predict(X_test)
y_pred_final = np.maximum(y_pred_final, 0)

print(f'MAE  : LKR {mean_absolute_error(y_test, y_pred_final):,.2f}')
print(f'RMSE : LKR {np.sqrt(mean_squared_error(y_test, y_pred_final)):,.2f}')
print(f'R2   : {r2_score(y_test, y_pred_final):.4f}')

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_final, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Cost (LKR)')
plt.ylabel('Predicted Cost (LKR)')
plt.title('Actual vs Predicted Construction Cost')
plt.tight_layout()
plt.savefig('cost_prediction_results.png')
print('Chart saved!')

import joblib
joblib.dump(final_model, 'construction_cost_model.pkl')
print('Best model saved!')


