# src/api.py

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model once
import mlflow.pyfunc

MODEL_URI = "runs:/02f2f188657d4d7f937ad70918cbbbfa/model"

model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def home():
    return {"message": "Credit Risk API running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # align missing columns
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in df:
            df[col] = 0

    df = df[model_features]

    pred = model.predict(df)

    # If pyfunc returns probability directly
    prob = float(pred[0])

    return {
        "default_probability": float(prob),
        "decision": "REJECT" if prob > 0.5 else "APPROVE"
    }