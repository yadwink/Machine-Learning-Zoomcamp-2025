# predict.py
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------- Load model artifact on startup ----------

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "models" / "xgb_stress_exercise.joblib"

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found at {ARTIFACT_PATH}. "
                            "Run train.py first to create it.")

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]
label_map = artifact["label_map"]  # e.g., {"STRESS": 0, "AEROBIC": 1, "ANAEROBIC": 2}
inv_label_map = {v: k for k, v in label_map.items()}


# ---------- Define input schema ----------

class InputFeatures(BaseModel):
    EDA_mean: float
    EDA_std: float
    TEMP_mean: float
    TEMP_std: float
    HR_mean: float
    HR_std: float
    BVP_mean: float
    BVP_std: float
    ACC_mag_mean: float
    ACC_mag_std: float


# ---------- Create FastAPI app ----------

app = FastAPI(
    title="Stress & Exercise Classifier",
    description="Classifies physiological sessions into STRESS, AEROBIC, or ANAEROBIC based on Empatica E4 features.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Stress/Exercise classifier is running.",
        "usage": "POST JSON to /predict with the required physiological features.",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "artifact_path": str(ARTIFACT_PATH),
        "n_features": len(feature_cols),
        "classes": list(label_map.keys()),
    }



@app.post("/predict")
def predict(features: InputFeatures) -> Dict:
    """
    Predicts the condition (STRESS, AEROBIC, ANAEROBIC) from engineered physiological features.
    """
    # Convert incoming data to a pandas DataFrame with the correct column order
    data_dict = features.model_dump()

    # Ensure the order matches feature_cols
    row = {col: data_dict[col] for col in feature_cols}
    df = pd.DataFrame([row], columns=feature_cols)

    # Predict class id
    pred_id = int(model.predict(df)[0])

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        proba_dict = {inv_label_map[i]: float(p) for i, p in enumerate(proba)}
    else:
        proba_dict = {}

    pred_label = inv_label_map[pred_id]

    return {
        "prediction": pred_label,
        "prediction_id": pred_id,
        "probabilities": proba_dict,
        "features_used": feature_cols,
    }
