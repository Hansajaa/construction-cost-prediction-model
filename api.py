from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("construction_cost_model.pkl")


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
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    prediction = max(float(prediction), 0.0)
    return {
        "estimated_cost_lkr": round(prediction, 2),
        "status": "success"
    }