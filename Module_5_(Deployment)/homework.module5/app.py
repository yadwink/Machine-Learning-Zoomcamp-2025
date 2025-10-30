from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

MODEL_PATH = Path("pipeline_v1.bin")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(client: Client):
    # DictVectorizer inside the pipeline expects a list of dicts
    features = [{
        "lead_source": client.lead_source,
        "number_of_courses_viewed": client.number_of_courses_viewed,
        "annual_income": client.annual_income,
    }]
    proba = float(model.predict_proba(features)[0, 1])
    return {"probability": proba}
