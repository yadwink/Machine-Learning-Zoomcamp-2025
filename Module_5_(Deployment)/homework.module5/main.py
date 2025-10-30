import pickle
from pathlib import Path

# ---------- config ----------
MODEL_PATH = Path("pipeline_v1.bin")

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
}
choices = [0.333, 0.533, 0.733, 0.933]
# ----------------------------

def pick_option(prob, options):
    # nearest; if exactly in-between, choose the higher option
    # (equivalent to rounding to nearest, ties go up)
    best = options[0]
    best_dist = abs(prob - best)
    for opt in options[1:]:
        dist = abs(prob - opt)
        if dist < best_dist or (abs(dist - best_dist) < 1e-12 and opt > best):
            best = opt
            best_dist = dist
    return best

def main():
    print("Loading model from:", MODEL_PATH.resolve())
    if not MODEL_PATH.exists():
        raise FileNotFoundError("pipeline_v1.bin not found in current folder")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # DictVectorizer inside the pipeline expects a list of dicts
    proba = model.predict_proba([record])[0, 1]
    print(f"Probability: {proba:.6f}")

    chosen = pick_option(proba, choices)
    print("Closest MCQ option:", chosen)

if __name__ == "__main__":
    main()
