import joblib
from pathlib import Path

import sys
import os

# Ensure the script can find its modules when run from any directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir.parent / "data" / "inventory.db"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # 1. Load Data
    df = load_vendor_invoice_data(db_path)

    # 2. Prepare Data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train Models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 4. Evaluate Models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    # 5. Select Best Model (based on lowest Mean Absolute Error)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model,
    }[best_model_name]

    # 6. Save Best Model for deployment
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()