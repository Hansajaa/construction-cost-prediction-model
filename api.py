from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("construction_cost_model.pkl")
risk_model = joblib.load("risk_classifier_model.pkl")

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


class TaskRiskInput(BaseModel):
    Task_Duration_Days: int
    Labor_Required: int
    Equipment_Units: int
    Material_Cost_LKR: float
    Start_Constraint: int
    Dependency_Count: int

@app.post("/predict-risk")
def predict_risk(data: TaskRiskInput):
    df = pd.DataFrame([data.dict()])
 
    predicted_class = risk_model.predict(df)[0]
 
    # predict_proba gives confidence for each class (Low/Medium/High)
    probabilities = risk_model.predict_proba(df)[0]
    class_labels = risk_model.classes_
 
    confidence_scores = {
        label: round(float(prob), 4)
        for label, prob in zip(class_labels, probabilities)
    }
 
    return {
        "predicted_risk_level": predicted_class,
        "confidence_scores": confidence_scores,
        "status": "success"
    }