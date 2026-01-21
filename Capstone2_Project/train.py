import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main(data_path: str, model_path: str):
    
    df = pd.read_csv('cleaned_dataset_Thyroid1.csv')

    # Split features/target
    X = df.drop("binaryClass", axis=1)
    y = df["binaryClass"]

    # Train/test split (same as notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Final model with optimal parameters
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    # Train on full training set
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    score = rmse(y_test, y_pred)
    print(f"Test RMSE: {score:.4f}")

    # Save model + feature names (IMPORTANT for prediction)
    artifact = {
        "model": model,
        "feature_names": list(X.columns)
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned CSV")
    parser.add_argument("--model-out", default="model.pkl", help="Output path for model pickle")
    args = parser.parse_args()

    main(args.data, args.model_out)
