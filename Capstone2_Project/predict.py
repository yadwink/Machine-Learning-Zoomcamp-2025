import pickle
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "model.pkl"

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_names = artifact["feature_names"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Build row with all expected features; missing ones default to 0
    row = {f: 0 for f in feature_names}
    row.update(data)

    X = pd.DataFrame([row], columns=feature_names)
    pred = float(model.predict(X)[0])

    return jsonify({"prediction": pred})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
