# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from prometheus_client import Counter, Gauge, generate_latest
from fastapi.responses import Response
import joblib
import os

# -------------------------
# App
# -------------------------
app = FastAPI()

# -------------------------
# Load model (LOCAL)
# -------------------------
MODEL_PATH = "artifacts/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Try to get feature names
try:
    model_features = model.feature_names_in_
except:
    model_features = None


# -------------------------
# Metrics
# -------------------------
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
PREDICTION_SCORE = Gauge("prediction_score", "Prediction score")


# -------------------------
# Input schema
# -------------------------
class InputData(BaseModel):
    data: dict


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk API running"}


@app.post("/predict")
def predict(input_data: InputData):

    REQUEST_COUNT.inc()

    df = pd.DataFrame([input_data.data])

    # Align columns
    if model_features is not None:
        for col in model_features:
            if col not in df:
                df[col] = 0
        df = df[model_features]

    # -------------------------
    # Prediction
    # -------------------------
    try:
        prob = float(model.predict_proba(df)[:, 1][0])
    except:
        # fallback if model doesn't support predict_proba
        prob = float(model.predict(df)[0])

    PREDICTION_SCORE.set(prob)

    return {
        "default_probability": prob,
        "decision": "REJECT" if prob > 0.5 else "APPROVE"
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")