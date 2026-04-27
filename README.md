# Invoice Intelligent System

## Project Objective
The **Invoice Intelligent System** is designed to **automate invoice processing, predict freight costs, and flag potentially incorrect or fraudulent invoices** using machine learning. It aims to **reduce manual effort, minimize human error, and provide actionable insights** for businesses handling large volumes of invoices and logistics data.

---

## Features
- **Invoice Verification & Flagging** – Automatically identifies suspicious or incorrect invoices.  
- **Freight Cost Prediction** – Predicts shipping and logistics costs accurately using ML models.  
- **End-to-End Automation** – Covers data preprocessing, model training, evaluation, and inference.  
- **User-Friendly Scripts** – Easy to run Python scripts for predictions without manual intervention.  
- **Data-Driven Insights** – Helps businesses make informed decisions quickly.  

---

## Benefits
- Saves time and reduces human errors in invoice management.  
- Helps detect frauds and discrepancies efficiently.  
- Enables predictive insights for logistics and freight planning.  
- Easy to maintain and scale with new models and data.  

---

## Project Structure

```
├── app.py # Main application entry point
├── freight_cost_prediction # Freight prediction scripts and models
│ ├── data_preprocessing.py
│ ├── model_evaluation.py
│ ├── train.py
│ └── models/predict_freight_model.pkl
├── invoice_flagging # Invoice flagging scripts and models
│ ├── data_preprocessing.py
│ ├── modeling_evaluation.py
│ ├── train.py
│ └── models/predict_flag_invoice.pkl
├── inference # Scripts for running predictions
├── notebooks # Jupyter notebooks for exploration & analysis
├── models # Saved ML models
├── data # Input datasets
├── README.md
└── TODO.md
```

---

## How to Use

1. **Clone the repository**:

```bash
git clone https://github.com/astha1504/Invoice-Intelligent-System.git
cd Invoice-Intelligent-System
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the main application**:

```bash
python app.py
```

4. **Run individual inference scripts (optional)**:

```bash
# Freight cost prediction
python inference/predict_freight.py

# Invoice flagging
python inference/predict_invoice_flag.py
```

## Flowchart

![Flowchart](flowchart.png)

**Flow Explanation:**

1. **Data Input** – Upload invoices and freight-related datasets.
2. **Data Preprocessing** – Clean, transform, and normalize the data.
3. **Model Training** – Train ML models for invoice flagging and freight cost prediction.
4. **Prediction / Flagging** – Generate predictions and flag suspicious invoices.
5. **Output** – Results available for analysis or export.

**Additional Notes**:
- Large files such as .pkl models and notebooks are tracked with Git LFS. Make sure Git LFS is installed if cloning the repository: `git lfs install`.
- `.ipynb_checkpoints/`, `.pyc` files, and databases (*.db) are excluded using `.gitignore`.

## Author

Astha Singh
B.Tech CSE (AI & ML) | AI Intern @ Infosys Springboard
GitHub: astha1504
