from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("src/models/xgb_model.joblib")
scaler = joblib.load("src/models/scaler.joblib")

class Transaction(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(transaction: Transaction):
    x = np.array(transaction.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[0, 1]
    pred = int(proba >= 0.5)
    return {"fraud_probability": float(proba), "prediction": pred}
