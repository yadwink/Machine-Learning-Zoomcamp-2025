# train.py
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Could not find features file at: {features_path}")
    df = pd.read_csv(features_path)
    required_cols = {"condition", "subject"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")
    return df


def build_X_y(df: pd.DataFrame):
    """
    Build feature matrix X and numeric labels y.

    Label mapping:
        STRESS   -> 0
        AEROBIC  -> 1
        ANAEROBIC-> 2
    """
    label_map = {"STRESS": 0, "AEROBIC": 1, "ANAEROBIC": 2}
    df = df.copy()
    df["label"] = df["condition"].map(label_map)

    if df["label"].isna().any():
        bad = df[df["label"].isna()]["condition"].unique()
        raise ValueError(f"Found unknown condition values: {bad}")

    feature_cols = [c for c in df.columns if c not in ["condition", "subject", "label"]]
    X = df[feature_cols]
    y = df["label"].astype(int).values

    return X, y, feature_cols, label_map


def train_xgboost(X, y, random_state: int = 42):
    """
    Train an XGBoost multi-class classifier with the same settings as in the notebook.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nXGBoost test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["STRESS", "AEROBIC", "ANAEROBIC"],
        )
    )

    return model


def main():
    project_root = Path(__file__).resolve().parent
    features_path = project_root / "data" / "processed" / "features_per_session.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_path}")
    df = load_features(features_path)

    print("Building X and y...")
    X, y, feature_cols, label_map = build_X_y(df)

    print("Training XGBoost model...")
    model = train_xgboost(X, y, random_state=42)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "label_map": label_map,
    }

    out_path = models_dir / "xgb_stress_exercise.joblib"
    joblib.dump(artifact, out_path)
    print(f"\n✅ Saved model artifact to: {out_path}")
    print("  - Contains: XGBoost model, feature column list, label mapping.")


if __name__ == "__main__":
    main()
