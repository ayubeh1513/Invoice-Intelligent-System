from modeling_evaluation import train_random_forest, evaluate_classifier

import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ----------------------------
# Define constants
# ----------------------------
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


# ----------------------------
# Helper Functions
# ----------------------------

def load_invoice_data():
    # Replace with your actual data path
    return pd.read_csv("data/invoice_data.csv")


def apply_labels(df):
    # Example labeling logic (modify as needed)
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled


# ----------------------------
# Main Pipeline
# ----------------------------

def main():

    # 1. Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # 2. Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'models/scaler.pkl'
    )

    # 3. Train model (GridSearchCV inside)
    grid_search = train_random_forest(X_train_scaled, y_train)

    # 4. Evaluate model
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # 5. Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        grid_search.best_estimator_,
        'models/predict_flag_invoice.pkl'
    )


# ----------------------------
# Run Script
# ----------------------------
if __name__ == "__main__":
    main()