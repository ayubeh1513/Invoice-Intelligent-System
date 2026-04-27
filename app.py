import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== LOAD MODEL ==============
BASE_DIR = Path(__file__).resolve().parent
INVOICE_MODEL_PATH = BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
INVOICE_SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"
FREIGHT_MODEL_PATH = BASE_DIR / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"

INVOICE_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

@st.cache_resource
def load_models():
    """Load models and scalers from pickle files"""
    models = {}
    
    # Load invoice flagging model
    try:
        if INVOICE_MODEL_PATH.exists() and INVOICE_SCALER_PATH.exists():
            models["invoice_model"] = joblib.load(INVOICE_MODEL_PATH)
            models["invoice_scaler"] = joblib.load(INVOICE_SCALER_PATH)
            logger.info("Invoice model and scaler loaded successfully")
        else:
            logger.warning("Invoice model files not found.")
    except Exception as e:
        logger.error(f"Error loading invoice model: {e}")

    # Load freight cost model
    try:
        if FREIGHT_MODEL_PATH.exists():
            models["freight_model"] = joblib.load(FREIGHT_MODEL_PATH)
            logger.info("Freight model loaded successfully")
        else:
            logger.warning("Freight model file not found.")
    except Exception as e:
        logger.error(f"Error loading freight model: {e}")
        
    return models

def predict_invoice_risk(data):
    """Make predictions on input data for invoice risk"""
    try:
        if "invoice_model" not in st.session_state.models:
            return "Model not loaded"
        
        model = st.session_state.models["invoice_model"]
        scaler = st.session_state.models["invoice_scaler"]
        
        if isinstance(data, pd.Series):
            data = data.to_frame().T
            
        # Ensure we only use the exact features required
        data = data[INVOICE_FEATURES]
        
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        return "High Risk" if prediction[0] == 1 else "Low Risk"
    except Exception as e:
        logger.error(f"Invoice prediction error: {e}")
        return f"Error: {str(e)}"

def predict_freight_cost(dollars):
    """Predict freight cost based on dollars"""
    try:
        if "freight_model" not in st.session_state.models:
            return "Model not loaded"
        
        model = st.session_state.models["freight_model"]
        df = pd.DataFrame([{"Dollars": dollars}])
        prediction = model.predict(df)
        return round(prediction[0], 2)
    except Exception as e:
        logger.error(f"Freight prediction error: {e}")
        return f"Error: {str(e)}"

def validate_invoice_inputs(inputs):
    """Validate input data for invoice risk"""
    try:
        if isinstance(inputs, pd.Series):
            # Check if all required features exist
            for feature in INVOICE_FEATURES:
                if feature not in inputs.index:
                    return False
            
            # Check for missing values in required features
            if inputs[INVOICE_FEATURES].isna().any():
                return False
            return True
        return False
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        return False

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = load_models()

# Page configuration
st.set_page_config(page_title='Invoice Intelligent System', layout='wide')

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select a Page', ['Dashboard', 'Invoice Risk Predictor', 'Freight Cost Predictor', 'Batch Upload (Invoice)'])

if page == 'Dashboard':
    st.title('Invoice Intelligent System Dashboard')
    st.markdown("Welcome to the Invoice Intelligent System. Use the sidebar to navigate to the predictor modules.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Invoice Risk Predictor: Evaluates if an invoice has potential risks or anomalies.")
    with col2:
        st.info("Freight Cost Predictor: Estimates freight cost based on the total dollar amount.")

elif page == 'Invoice Risk Predictor':
    st.title('Invoice Risk Predictor')
    st.markdown("Enter the invoice details below to predict if there is a risk.")
    
    with st.form("invoice_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            inv_qty = st.number_input("Invoice Quantity", min_value=0.0, value=100.0)
            inv_dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=500.0)
            freight = st.number_input("Freight Cost ($)", min_value=0.0, value=50.0)
            
        with col2:
            tot_item_qty = st.number_input("Total Item Quantity (PO)", min_value=0.0, value=100.0)
            tot_item_dollars = st.number_input("Total Item Dollars (PO) ($)", min_value=0.0, value=500.0)
            
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            input_data = pd.Series({
                "invoice_quantity": inv_qty,
                "invoice_dollars": inv_dollars,
                "Freight": freight,
                "total_item_quantity": tot_item_qty,
                "total_item_dollars": tot_item_dollars
            })
            
            if validate_invoice_inputs(input_data):
                prediction = predict_invoice_risk(input_data)
                if prediction == "High Risk":
                    st.error(f'Prediction: {prediction}')
                elif prediction == "Low Risk":
                    st.success(f'Prediction: {prediction}')
                else:
                    st.warning(f'Result: {prediction}')
            else:
                st.error('Invalid inputs. Please check your data.')

elif page == 'Freight Cost Predictor':
    st.title('Freight Cost Predictor')
    st.markdown("Estimate the freight cost based on purchase dollars.")
    
    with st.form("freight_form"):
        dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=1000.0)
        submitted = st.form_submit_button("Predict Freight Cost")
        
        if submitted:
            prediction = predict_freight_cost(dollars)
            if isinstance(prediction, (int, float)):
                st.success(f'Predicted Freight Cost: ${prediction}')
            else:
                st.error(f'Result: {prediction}')

elif page == 'Batch Upload (Invoice)':
    st.title('Batch Upload - Invoice Risk')
    st.markdown("Upload a CSV file containing invoice data to predict risks in bulk.")
    st.markdown(f"**Required columns:** `{', '.join(INVOICE_FEATURES)}`")
    
    uploaded_file = st.file_uploader('Upload CSV file', type='csv')
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            results = []
            
            # Check if required columns exist
            missing_cols = [col for col in INVOICE_FEATURES if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            else:
                for index, row in df.iterrows():
                    if validate_invoice_inputs(row):
                        prediction = predict_invoice_risk(row)
                        results.append(prediction)
                    else:
                        results.append('Invalid input (Missing values)')
                
                # Add predictions to original dataframe
                df['Risk_Prediction'] = results
                
                st.write("Preview of results:")
                st.dataframe(df.head(10))
                
                st.download_button(
                    label='Download Results CSV',
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='invoice_risk_results.csv',
                    mime='text/csv',
                    key='download_results'
                )
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            st.error(f'Error processing the uploaded file: {str(e)}')
