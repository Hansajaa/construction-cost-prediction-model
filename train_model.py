import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (CSV, Excel, or DB export)
df = pd.read_csv("construction_data.csv")

# Explore
print(df.shape)         # Rows and columns
print(df.head())        # First 5 rows
print(df.info())        # Data types
print(df.describe())    # Statistical summary
print(df.isnull().sum()) # Check for missing values

# Drop duplicates
df.drop_duplicates(inplace=True)

df['total_area_sqft'] = df['total_area_sqft'].fillna(df['total_area_sqft'].median())
df['project_type'] = df['project_type'].fillna(df['project_type'].mode()[0])

# Remove outliers using IQR method
Q1 = df['total_cost'].quantile(0.25)
Q3 = df['total_cost'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['total_cost'] >= Q1 - 1.5*IQR) & (df['total_cost'] <= Q3 + 1.5*IQR)]

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Define feature columns
categorical_features = ['project_type', 'location_zone', 'material_grade', 'soil_type']
numerical_features = ['total_area_sqft', 'num_floors', 'labor_cost_per_day',
                      'project_duration_days', 'num_workers']

# Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

from sklearn.model_selection import train_test_split

X = df.drop('total_cost', axis=1)   # Features
y = df['total_cost']                 # Target (cost to predict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Build full pipeline (preprocessing + model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "R2": r2, "pipeline": pipeline}
    print(f"{name:25s} → MAE: ${mae:,.2f}  |  R²: {r2:.4f}")

    from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV

# ✅ Tune Gradient Boosting (best performer)
param_grid = {
    'model__n_estimators':    [100, 200],
    'model__max_depth':       [3, 5, 7],
    'model__learning_rate':   [0.05, 0.1, 0.2],
    'model__min_samples_split': [2, 5]
}

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))  # ✅ Changed
])

grid_search = GridSearchCV(best_pipeline, param_grid, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
final_model = grid_search.best_estimator_

# Evaluate
y_pred_final = final_model.predict(X_test)

print(f"MAE  : LKR {mean_absolute_error(y_test, y_pred_final):,.2f}")
print(f"RMSE : LKR {np.sqrt(mean_squared_error(y_test, y_pred_final)):,.2f}")
print(f"R²   : {r2_score(y_test, y_pred_final):.4f}")

# Visualize predictions vs actuals
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_final, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cost (LKR)")
plt.ylabel("Predicted Cost (LKR)")
plt.title("Actual vs Predicted Construction Cost")
plt.tight_layout()
plt.savefig("cost_prediction_results.png")
print("📊 Chart saved!")

# Save
import joblib
joblib.dump(final_model, "construction_cost_model.pkl")
print("✅ Best model saved!")