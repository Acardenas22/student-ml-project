# train_model.py
"""
Train a simple regression pipeline on the Wine Quality (red) dataset.
Saves the best model (by RMSE on test set) and the scaler to ./model/
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Paths
DATA_PATH = os.path.join("data", "winequality-red.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    return df

def prepare_data(df, target_col="quality", test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return scaler, X_train_s, X_test_s

def train_and_evaluate(X_train, y_train, X_test, y_test):
    results = {}

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42, max_iter=10000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "rmse": rmse,
            "r2": r2
        }

        print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return results

def pick_best_model(results):
    # Choose model with lowest RMSE
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    return best_name, results[best_name]["model"], results[best_name]

def save_artifacts(model, scaler, model_dir=MODEL_DIR, model_name="model.joblib", scaler_name="scaler.joblib"):
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, scaler_name)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    return model_path, scaler_path

def main():
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}")

    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Scaling data...")
    scaler, X_train_s, X_test_s = scale_data(X_train, X_test)

    print("Training and evaluating models...")
    results = train_and_evaluate(X_train_s, y_train, X_test_s, y_test)

    best_name, best_model, best_stats = pick_best_model(results)
    print(f"Best model: {best_name} with RMSE={best_stats['rmse']:.4f} R2={best_stats['r2']:.4f}")

    print("Saving artifacts...")
    save_artifacts(best_model, scaler)

    print("Done.")

if __name__ == "__main__":
    main()