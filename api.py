from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("construction_cost_model.pkl")

# Define input structure
class ProjectInput(BaseModel):
    project_type: str
    total_area_sqft: float
    num_floors: int
    location_zone: str
    material_grade: str
    labor_cost_per_day: float
    project_duration_days: int
    soil_type: str
    num_workers: int

@app.post("/predict-cost")
def predict_cost(data: ProjectInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {
        "estimated_cost_lkr": round(prediction, 2),
        "status": "success"
    }