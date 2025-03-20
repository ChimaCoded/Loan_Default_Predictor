# Loan Default Predictor

## Introduction

This project builds a machine learning solution to predict loan default risk using loan data (`loan_data.csv`). The goal is to predict if an applicant is likely to default (1) or pay in full (0) based on various financial and loan-specific features. The solution includes data preprocessing, feature engineering, model selection, training, tuning, interpretability via SHAP, and a FastAPI deployment for real-time predictions. 

Key features:
- **Data Preprocessing**: Handles missing values, outliers, and inconsistencies.
- **Feature Engineering**: Adds derived features like `monthly_payment` and `total_loan_burden`.
- **Model**: Selects between XGBoost and RandomForest based on performance with hyperparameter tuning via `RandomizedSearchCV`.
- **Interpretability**: SHAP values explain predictions and feature importance.
- **Deployment**: FastAPI API with endpoints for prediction, risk levels, and insights.

---

## Setup and Running the Project

### Prerequisites
- **Python**: 3.11.6 
- **Dependencies**: Listed in `requirements.txt`
- **Ensure data/loan_data.csv is in the project root.**


### Installation
1. **Clone the Repository**:

   git clone https://github.com/ChimaCoded/Loan_Default_Predictor.git
   cd loan_default_predictor

2. **Create Virtual Environment**:

python -m venv venv
.\venv\Scripts\activate.bat  # for Windows users
source venv/bin/activate  # Linux/Mac users

3. **Install Dependencies**:

pip install -r requirements.txt

4. **Running the App**:

python main.py

This trains the model, saves it to models/model.pkl, and starts the FastAPI server.

Server starts at http://0.0.0.0:8000 

5. **Accessing the API**: 
URL: http://0.0.0.0:8000

Endpoints: See "Example Payloads" below.

Test: Use Postman or curl to hit endpoints.

Example

1. /predict
Predicts if a loan will default.
Request: POST http://0.0.0.0:8000/predict

Payload:
{
  "loan_amount": 10000.0,
  "term": 36,
  "interest_rate": 5.0,
  "monthly_income": 3000.0,
  "credit_score": 700,
  "employment_status": "Employed",
  "loan_purpose": "Personal",
  "num_previous_loans": 2,
  "default_history": 0
}

2.  /risk_level
Assigns a risk category (Low, Medium, High) 
Request: POST http://0.0.0.0:8000/risk_level

Payload: Same as above.

3. /interpretability
Provides SHAP-based feature contributions.
Request: POST http://0.0.0.0:8000/interpretability

Payload: Same as above.


4. /feature_importance
Lists global feature importance.
Request: GET http://0.0.0.0:8000/feature_importance


**File Structure**

loan_default_predictor/
├── data/
│   └── loan_data.csv        # Dataset (500 rows, 12 columns)
├── models/
│   ├── model.pkl           # Trained model
│   └── preprocessor.pkl    # Preprocessing pipeline
├── app/
│   ├── __init__.py
│   ├── api.py             # FastAPI app, endpoints
│   ├── data_handler.py    # Data loading, cleaning, preprocessing
│   └── model_handler.py   # Model training, tuning, SHAP
├── main.py                # Entry point, runs Uvicorn
├── requirements.txt       # Dependencies
└── README.md              # readme file








