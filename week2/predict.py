# predict.py
"""
Quick smoke-test for saved model artifacts:
- loads model/model.joblib + model/scaler.joblib
- loads the dataset
- predicts one sample and prints results
"""

import os
import joblib
import pandas as pd
import numpy as np

DATA_PATH = os.path.join("data", "winequality-red.csv")
MODEL_PATH = os.path.join("model", "model.joblib")
SCALER_PATH = os.path.join("model", "scaler.joblib")

def main():
    # Load artifacts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Predict a few random samples
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=5, replace=False)

    print("=== Prediction Smoke Test (5 random rows) ===")
    for i in indices:
        x_raw = X.iloc[[i]]
        y_true = y.iloc[i]

        x_scaled = scaler.transform(x_raw)
        y_pred = model.predict(x_scaled)[0]

        print(
            f"Row {i:4d} | true={y_true} | "
            f"pred={y_pred:.3f} | rounded={int(np.round(y_pred))}"
        )

if __name__ == "__main__":
    main()