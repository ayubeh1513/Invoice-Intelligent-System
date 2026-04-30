# Vendor Invoice Intelligence System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

An end-to-end AI & ML system built with Python and Streamlit to **automate invoice verification, predict freight costs, and flag high-risk invoices in real time** — replacing slow, error-prone manual audits with intelligent automation.

---

### Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Workflow](#workflow)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
6. [Model Selection & Performance](#model-selection--performance)
7. [Results & Business Impact](#results--business-impact)
8. [Technology Stack](#technology-stack)
9. [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Running the App](#running-the-app)
10. [Project Structure](#project-structure)
11. [Future Scope](#future-scope)
12. [Contributing](#contributing)
13. [Author](#author)
14. [License](#license)

---

## Introduction

The **Vendor Invoice Intelligence System** is designed to replace manual, time-consuming invoice audits with a transparent, intelligent, and automated alternative. Businesses receive thousands of vendor invoices daily — invoice discrepancies and freight cost overcharges cost companies **1–5% of total procurement spend**, and audit teams struggle to keep up.

This system tackles the problem end-to-end: from raw SQLite data and SQL-based feature engineering, through model training and evaluation, to a deployed **Streamlit web application** that any business user can operate — with no ML expertise required.

The project is built around **two core ML tasks**:
- 🔮 **Freight Cost Prediction** — A regression model that accurately estimates the expected freight charge for any invoice.
- 🚨 **Invoice Risk Flagging** — A binary classification model that identifies suspicious or high-risk invoices before payment is processed.

## Key Features

* ✅ **Invoice Verification & Flagging** — Automatically identifies suspicious or fraudulent invoices using a trained Random Forest Classifier.
* ✅ **Freight Cost Prediction** — Predicts shipping and logistics costs with 91.6% R² accuracy using a Random Forest Regressor.
* ✅ **SQL Feature Engineering** — 6 meaningful derived features extracted from 4 relational database tables via SQL queries.
* ✅ **End-to-End Pipeline** — Covers data loading, preprocessing, model training, evaluation, inference, and deployment in one codebase.
* ✅ **Interactive Streamlit App** — Dashboard, single-invoice prediction, and batch CSV upload pages in one clean UI.
* ✅ **Batch Processing** — Upload a CSV of multiple invoices and download a results file with risk flags instantly.
* ✅ **Model Serialization** — Trained models saved as `.pkl` files for fast, reproducible inference without retraining.
* ✅ **Data-Driven Insights** — Feature importance and EDA findings surface which invoice attributes drive cost and risk the most.

## Workflow

The system follows a clear, automated five-step pipeline:

1. **Data Input** — Load vendor invoice and purchase records from the SQLite database (`inventory.db`).
2. **Preprocessing & Feature Engineering** — SQL-derived features are computed; data is cleaned, joined, labeled, and scaled.
3. **Model Training** — Random Forest models are trained for regression (freight cost) and classification (risk flag) with GridSearchCV tuning.
4. **Prediction & Flagging** — The trained models predict freight costs and classify invoices as `High Risk` or `Normal` in real time.
5. **Output** — Results are displayed in the Streamlit dashboard and can be exported as a downloadable CSV.

```
inventory.db (SQLite)
      │
      ▼
SQL Feature Engineering  ──►  6 Derived Features
      │
      ▼
Python Preprocessing Pipeline
  ├─ JOIN Aggregation (vendor_invoice + purchases on PONumber)
  ├─ Risk Label Creation (dollar discrepancy + receiving delay)
  ├─ 80 / 20 Train-Test Split (stratified, random_state=42)
  └─ StandardScaler (fit on train, transform on test → saved as scaler.pkl)
      │
      ▼
Model Training
  ├─ Task 1 → Random Forest Regressor  →  predict_freight_model.pkl
  └─ Task 2 → Random Forest Classifier →  predict_flag_invoice.pkl
      │
      ▼
Streamlit Web App (app.py)
  ├─ Dashboard
  ├─ Predict  (single invoice)
  └─ Batch Upload  (CSV in → results CSV out)
```

## Dataset Description

**Source:** SQLite database — `inventory.db`

| Table | Key Columns | Purpose |
|---|---|---|
| `vendor_invoice` | PONumber, VendorNumber, Quantity, Dollars, Freight, PODate, InvoiceDate, PayDate | Core invoice records — primary source for both ML tasks |
| `purchases` | PONumber, Brand, Quantity, Dollars, ReceivingDate | Purchase-level line items per PO used to compute aggregated totals |
| `purchase_prices` | Brand, PurchasePrice, Volume, Classification, PricePerBottle | Reference prices used for cost anomaly analysis |
| `inventory tables` | Store, City, Zip, State, Category, Description | Geographic and product-level context |

**Task mapping:**
- `vendor_invoice` → **Regression** (predict `Freight`) and **Classification** (flag risky invoices)
- `purchases` → joined to compute aggregate dollar and quantity totals per PO

## Data Preprocessing & Feature Engineering

### SQL-Derived Features

| Feature | Description |
|---|---|
| `days_po_to_invoice` | Days from PO creation date to invoice date |
| `days_to_pay` | Days from invoice to payment — acts as a liquidity indicator |
| `total_brands` | Count of distinct brands per purchase order |
| `total_item_quantity` | Aggregated item quantities from the purchases table |
| `total_item_dollars` | Sum of purchase-level dollar amounts per PO |
| `avg_receiving_delay` | Average delay between PO date and the receiving date |

### Python Pipeline Steps

| Step | Detail |
|---|---|
| 🗄️ Data Loading | `pandas.read_sql_query()` loads `vendor_invoice` and `purchases` from SQLite |
| 🔗 JOIN Aggregation | `LEFT JOIN` of `vendor_invoice` with purchase-level aggregates on `PONumber` |
| 🏷️ Risk Labeling | `flag = 1` if `\|invoice_$ − item_$\| > 5` **OR** `avg_receiving_delay > 10 days` |
| ✂️ Train/Test Split | 80/20 stratified split with `random_state=42` for reproducibility |
| ⚖️ Feature Scaling | `StandardScaler` fit on train set, applied to test set; saved as `scaler.pkl` |

### Key EDA Findings

- ~35% of invoices show a dollar discrepancy of > $5 compared to purchase records.
- Invoices with `avg_receiving_delay > 10 days` are **2.4× more likely** to be flagged as risky.
- Freight cost has a strong positive correlation with invoice dollar amount (**r ≈ 0.87**).
- A T-test confirms a statistically significant mean difference between flagged and normal invoices (**p < 0.01**).

## Model Selection & Performance

### Task 1 — Freight Cost Prediction (Regression)

| Model | MAE | RMSE | R² Score | Selected |
|---|---|---|---|---|
| Linear Regression | $12.45 | $18.92 | 74.2% | — |
| Decision Tree Regressor | $9.87 | $15.34 | 83.1% | — |
| **Random Forest Regressor** | **$7.23** | **$11.18** | **91.6%** | ✅ Best |

> Random Forest outperforms the linear baseline by **17.4% on R²**, confirming the freight–dollar relationship is non-linear.

### Task 2 — Invoice Risk Flagging (Classification)

| Metric | Score | Interpretation |
|---|---|---|
| Accuracy | 94.8% | 94.8% of all invoices correctly classified |
| Precision | 92.3% | Only 7.7% false-positive rate — minimises unnecessary invoice holds |
| Recall | 88.7% | Catches ~89 of every 100 truly risky invoices |
| F1-Score | 90.5% | Balanced performance across imbalanced classes |

**Tuning:** GridSearchCV with 5-fold cross-validation, optimising for F1-Score.  
**Features used:** `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`.

## Results & Business Impact

| Metric | Value |
|---|---|
| 🎯 R² Score (Regression) | **91.6%** |
| 💰 Mean Absolute Error | **$7.23** |
| ✅ Classification Accuracy | **94.8%** |
| ⚖️ F1-Score | **90.5%** |

- **`invoice_dollars`** and **`total_item_dollars`** are the top-2 most important features in both models.
- Models are serialised as `.pkl` files and served via Streamlit for real-time single or batch inference.
- Early detection of fraudulent and erroneous invoices provides **direct, measurable ROI** by reducing cost leakage before payments are processed.

## Technology Stack

* **Application:** Streamlit
* **Language:** Python 3.9+
* **ML Framework:** Scikit-learn
    * Random Forest Regressor & Classifier
    * GridSearchCV, StandardScaler, train_test_split
* **Data Layer:** SQLite + Pandas (`read_sql_query`)
* **Serialisation:** Joblib (`.pkl` model files)
* **Notebooks:** Jupyter (for EDA and experimentation)

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9 or higher
* `pip` package manager
* Git (and optionally **Git LFS** — see note below)
* A SQLite database file (`inventory.db`) with the required tables

> ⚠️ **Git LFS Note:** Large files such as `.pkl` models and Jupyter notebooks are tracked with Git LFS. Run `git lfs install` before cloning to ensure they download correctly.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayubeh1513/Invoice-Intelligent-System
   cd Invoice-Intelligent-System
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # On macOS / Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Core packages installed:

   | Package | Minimum Version |
   |---|---|
   | streamlit | ≥ 1.28.0 |
   | scikit-learn | ≥ 1.3.0 |
   | pandas | ≥ 2.0.0 |
   | numpy | ≥ 1.24.0 |
   | joblib | ≥ 1.3.0 |

4. **(Optional) Retrain the models on your own data:**

   Place your `inventory.db` in the project root, then run:
   ```bash
   python freight_cost_prediction/train.py
   python invoice_flagging/train.py
   ```
   This regenerates the `.pkl` model and scaler files.

### Running the App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

**Run individual inference scripts (optional):**
```bash
# Freight cost prediction
python inference/predict_freight.py

# Invoice risk flagging
python inference/predict_invoice_flag.py
```

The Streamlit app has three pages accessible via the sidebar:

| Page | Description |
|---|---|
| 📊 Dashboard | Overview of key system metrics and model performance summary |
| 🔮 Predict | Enter features for a single invoice and get an instant risk prediction |
| 📂 Batch Upload | Upload a CSV of invoices; download `results.csv` with predictions for all rows |

---

## Project Structure

```
Invoice-Intelligent-System/
│
├── app.py                              # Main Streamlit application entry point
├── requirements.txt                    # Python dependencies
├── README.md
├── TODO.md
│
├── freight_cost_prediction/            # Regression pipeline — freight cost
│   ├── data_preprocessing.py           # Load & prepare features from SQLite
│   ├── train.py                        # Model training script
│   ├── model_evaluation.py             # Evaluation metrics & reporting
│   └── models/
│       └── predict_freight_model.pkl   # Trained Random Forest Regressor
│
├── invoice_flagging/                   # Classification pipeline — risk flagging
│   ├── data_preprocessing.py           # Feature engineering + risk labeling
│   ├── train.py                        # Model training script
│   ├── modeling_evaluation.py          # Evaluation metrics & reporting
│   └── models/
│       ├── predict_flag_invoice.pkl    # Trained Random Forest Classifier
│       └── scaler.pkl                  # Fitted StandardScaler
│
├── inference/                          # Standalone inference scripts
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── models/                             # Top-level model copies (used by app.py)
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
│
└── notebooks/                          # Jupyter notebooks for EDA & experimentation
    ├── invoice_flagging.ipynb
    └── predicting freight cost.ipynb
```

> **Note:** `.ipynb_checkpoints/`, `.pyc` files, and `*.db` databases are excluded via `.gitignore`.

---

## Future Scope

| Roadmap Item | Description |
|---|---|
| 🔍 NLP-based Invoice Parsing | Use OCR + NLP to extract structured data directly from PDF or image invoices |
| ⏱️ Real-Time Stream Processing | Integrate Apache Kafka for live invoice monitoring and instant alerting |
| 🧠 Deep Learning Models | Explore LSTM / Transformer architectures for sequential invoice pattern detection |
| 📈 Anomaly Detection | Unsupervised models (e.g. Isolation Forest) to catch novel or previously unseen fraud patterns |
| 🔗 ERP Integration | Connect with SAP / Oracle ERP systems for end-to-end procurement automation |

---

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Author

**Ayushman Behera**  
B.Tech CSE (DSML)
Lovely Professional University, Jalandhar, Punjab  
🔗 GitHub: [ayubeh1513](https://github.com/ayubeh1513)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
