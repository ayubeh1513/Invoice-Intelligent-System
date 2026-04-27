import joblib
import pandas as pd
import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model and scaler paths
MODEL_PATH = os.path.join(BASE_DIR, "invoice_flagging", "models", "predict_flag_invoice.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "invoice_flagging", "models", "scaler.pkl")

# Required features for prediction
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

def load_model_and_scaler(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """
    Load trained invoice flagging model and scaler.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler not found at: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_invoice_risk(input_data):
    """
    Predict risk flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict or list of dicts

    Returns
    -------
    pd.DataFrame with predicted risk flag
    """
    if isinstance(input_data, dict):
        input_data = [input_data]
        
    if not isinstance(input_data, list):
        raise ValueError("❌ input_data must be a dictionary or a list of dictionaries.")

    input_df = pd.DataFrame(input_data)

    # Check for missing required features
    missing_features = [f for f in FEATURES if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing required features: {missing_features}")

    # Ensure numeric types and no NaNs
    try:
        input_df[FEATURES] = input_df[FEATURES].apply(pd.to_numeric, errors="raise")
    except Exception as e:
        raise ValueError("❌ All required features must contain only numeric values.") from e

    if input_df[FEATURES].isna().any().any():
        raise ValueError("❌ Input features contain missing values (NaN). Please provide valid numbers.")

    model, scaler = load_model_and_scaler()
    
    # Scale data
    X_scaled = scaler.transform(input_df[FEATURES])
    
    # Predict
    predictions = model.predict(X_scaled)
    input_df['Risk_Flag'] = predictions
    input_df['Risk_Label'] = input_df['Risk_Flag'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
    
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = [
        {
            "invoice_quantity": 100,
            "invoice_dollars": 500,
            "Freight": 50,
            "total_item_quantity": 100,
            "total_item_dollars": 500
        },
        {
            "invoice_quantity": 50,
            "invoice_dollars": 1000,
            "Freight": 120,
            "total_item_quantity": 40,
            "total_item_dollars": 500
        }
    ]
    
    try:
        prediction_df = predict_invoice_risk(sample_data)
        print("Predictions:")
        print(prediction_df[['Risk_Flag', 'Risk_Label']])
    except Exception as e:
        print(f"Error during prediction: {e}")
