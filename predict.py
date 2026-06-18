import joblib
import pandas as pd

# Load saved model
model = joblib.load("construction_cost_model.pkl")

# Test with a new project
new_project = pd.DataFrame([{
    "project_type":           "Commercial",
    "total_area_sqft":        8000,
    "num_floors":             10,
    "location_zone":          "Urban",
    "material_grade":         "Standard",
    "labor_cost_per_day":     6000,
    "project_duration_days":  340,
    "soil_type":              "Clay",
    "num_workers":            50
}])

predicted_cost = model.predict(new_project)
print(f"💰 Estimated Cost: LKR {predicted_cost[0]:,.2f}")