import sys
import os
import joblib

# Ensure the script can find its modules when run from any directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features,
)




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
# Main Pipeline
# ----------------------------

def main():

    # 1. Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # 2. Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scaler.pkl')
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
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(
        grid_search.best_estimator_,
        os.path.join(models_dir, 'predict_flag_invoice.pkl')
    )


# ----------------------------
# Run Script
# ----------------------------
if __name__ == "__main__":
    main()