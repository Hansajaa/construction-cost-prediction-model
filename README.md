# 🏗️ Construction Cost Prediction — AI/ML Model

An AI-driven construction cost prediction model built with **scikit-learn** and served via a **FastAPI** REST API. Designed to integrate with a Construction Management System (CMS).

---

## 📁 Project Structure

```
construction_project/
├── construction_data.csv           # Dataset (your project data)
├── train_model.py                  # Model training script
├── predict.py                      # Quick local prediction test
├── api.py                          # FastAPI REST API
├── requirements.txt                # Python dependencies
├── construction_cost_model.pkl     # Saved trained model (auto-generated)
├── cost_prediction_results.png     # Evaluation chart (auto-generated)
└── README.md                       # This file
```

---

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | ML models (Gradient Boosting, Random Forest, Linear Regression) |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Data visualization |
| joblib | Save and load trained model |
| FastAPI | REST API to serve predictions |
| uvicorn | ASGI server to run FastAPI |

---

## 🚀 Getting Started from Scratch

### 1. Clone the Repository

```bash
git clone https://github.com/Hansajaa/construction-cost-prediction-model.git
cd construction-cost-prediction-model
```

### 2. Create a Virtual Environment

```bash
python -m venv construction_ai_env
```

Activate it:

```bash
# Windows
construction_ai_env\Scripts\activate

# macOS / Linux
source construction_ai_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Dataset

Place your `construction_data.csv` file in the root project folder.

Your CSV must have these columns:

| Column | Type | Example |
|---|---|---|
| project_type | text | Residential, Commercial, Industrial |
| total_area_sqft | number | 8000 |
| num_floors | number | 5 |
| location_zone | text | Urban, Suburban, Rural |
| material_grade | text | Economy, Standard, Premium |
| labor_cost_per_day | number | 6000 |
| project_duration_days | number | 240 |
| soil_type | text | Clay, Sandy, Rocky, Expansive |
| num_workers | number | 40 |
| total_cost | number | 750000000 |

### 5. Train the Model

```bash
python train_model.py
```

This will:
- Clean and preprocess the dataset
- Train and compare 3 ML models
- Tune the best model (Gradient Boosting) using GridSearchCV
- Print MAE, RMSE, and R² scores
- Save the trained model as `construction_cost_model.pkl`
- Save the evaluation chart as `cost_prediction_results.png`

### 6. Test a Prediction Locally

```bash
python predict.py
```

You should see output like:
```
💰 Estimated Cost: LKR 742,500,000.00
```

### 7. Run the API

```bash
python -m uvicorn api:app --reload
```

The API will be available at:
```
http://localhost:8000
```

Open the interactive API docs at:
```
http://localhost:8000/docs
```

---

## 📡 API Usage

### Endpoint

```
POST /predict-cost
```

### Request Body

```json
{
  "project_type": "Commercial",
  "total_area_sqft": 8000,
  "num_floors": 5,
  "location_zone": "Urban",
  "material_grade": "Standard",
  "labor_cost_per_day": 6000,
  "project_duration_days": 240,
  "soil_type": "Clay",
  "num_workers": 40
}
```

### Response

```json
{
  "estimated_cost_lkr": 742500000.00,
  "status": "success"
}
```

### Example using JavaScript (for CMS frontend)

```javascript
const response = await fetch("http://localhost:8000/predict-cost", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    project_type: "Residential",
    total_area_sqft: 5000,
    num_floors: 3,
    location_zone: "Urban",
    material_grade: "Standard",
    labor_cost_per_day: 5000,
    project_duration_days: 180,
    soil_type: "Sandy",
    num_workers: 25
  })
});

const data = await response.json();
console.log("Predicted Cost:", data.estimated_cost_lkr);
```

---

## 📊 Model Performance

| Model | MAE | R² |
|---|---|---|
| Linear Regression | ~137M LKR | ~0.78 |
| Random Forest | ~148M LKR | ~0.74 |
| **Gradient Boosting** | **~120M LKR** | **~0.83** |

> ✅ **Gradient Boosting** was selected as the final model after hyperparameter tuning with GridSearchCV.

---

## 📦 requirements.txt

```
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
fastapi
uvicorn
```

---

## 🔄 How to Retrain the Model

If you update your dataset, simply rerun:

```bash
python train_model.py
```

The old `construction_cost_model.pkl` will be overwritten with the new trained model.

---

## 📌 Notes

- Dataset should have at least **500+ rows** for reliable predictions
- R² score above **0.85** is recommended before production use
- All costs are in **Sri Lankan Rupees (LKR)**
- The model is saved as a **scikit-learn Pipeline** — preprocessing is included inside the `.pkl` file, so no manual scaling is needed during prediction
