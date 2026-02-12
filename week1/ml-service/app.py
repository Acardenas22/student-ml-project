import os
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import SuggestReq, SuggestResp
import ml_logic

app = FastAPI(title="AAD ML Service (Week 1)")

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

FEATURE_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = os.path.join("model", "model.joblib")
SCALER_PATH = os.path.join("model", "scaler.joblib")

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    # API can still start; /predict will raise a clean error
    print(f"[WARN] Could not load model artifacts: {e}")

# CORS: permissive for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ml/suggest", response_model=SuggestResp)
async def suggest(req: SuggestReq):
    """
    Week 1 endpoint: template-based suggestions.
    Later weeks can replace ml_logic internals with real models.
    """
    try:
        suggestions, notes = ml_logic.generate_suggestions(req)
        return SuggestResp(suggestions=suggestions, notes=notes)

    except Exception as e:
        # Convert internal errors into a clean HTTP error
        raise HTTPException(
            status_code=500,
            detail=f"ML service error: {str(e)}"
        )

@app.post("/predict")
async def predict(features: WineFeatures):

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Check model/ folder.")
    # Map JSON fields -> training column names (with spaces)
    X = pd.DataFrame([{
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol,
    }], columns=FEATURE_COLS)

    Xs = scaler.transform(X)
    pred = float(model.predict(Xs)[0])

    return {
        "predicted_quality": pred,
        "predicted_quality_rounded": int(np.round(pred))
    }