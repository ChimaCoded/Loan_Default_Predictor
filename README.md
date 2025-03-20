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

**Example Payloads**

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

![predict](https://github.com/user-attachments/assets/613eda09-93e1-41c8-9f97-969c928df58d)

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

![folder structure](https://github.com/user-attachments/assets/60860b92-17da-493d-9692-bd0d4bd44a00)


## Approach and Rationale
**Data Preprocessing & Feature Engineering**
Cleaning: There were no missing values for the 500-row loan dataset. However, checks were made available for sake of scalability. Also checked for inconsistent data.

Preprocessing: Applied StandardScaler to numerical features and OneHotEncoder to categorical (employment_status, loan_purpose) to ensures te data is compatible with the models.

Feature Engineering: After understanding the dataset, some new data was derived for more feature engineering:

- Monthly Payment: Derived using (loan_amount * monthly_rate * (1 + monthly_rate)^term) / ((1 + monthly_rate)^term - 1).

- Debt-to-Income Ratio: Computed as monthly_payment / monthly_income. 

- Total Loan Burden: Defined as loan_amount * (num_previous_loans + 1)—measures cumulative debt.

Justification: These engineered features helps to enhance predictive power.

**Model Selection**
XGBoost and RandomForest were chosen over models like logistic regression or even neural networks for their ability to handle the small datasets, non-linear patterns and dataa imbalance efficiently. 

XGBoost was selected over RandomForest because of it's gradient boosting which showed higher AUC-ROC with scale_pos_weight, and optimised via RandomizedSearchCV.


**Performance Evaluation**
Metrics:
Final Test Set Metrics: {'roc_auc': 0.6327, 'precision': 0.3529, 'recall': 0.4444, 'f1': 0.3934}

ROC-AUC was Chosen becuase it performed and ranked higher than the other metrics across classes.


**Interpretability & Business Insights**
SHAP Analysis:

![shap_feature_importance](https://github.com/user-attachments/assets/32bf4bd6-62b3-4dda-ad04-ae386c4922d3)

This shows the various importance of the features in the performance of the model. 

**Deployment Readiness**
FastAPI: Asynchronous API with various endpoints (/predict, /risk_level, /interpretability, /feature_importance) cover all use cases. It can also be customized to add more endpoints for scalability. 

Logging: INFO-level logs to console to track behavior cleanly.

**Additional Insights**
Challenges: The dataset of 500 rows and 12 columns is quite small and risks overfitting. Also allowed libraries to use limits the extent of using advanced algotrithms for prediction.  

Future Enhancements: 

- SMOTE for imbalance
- More EDA diagrams for visualizations
- Database integration for real-time updates,
- Cloud hosting to help for scalabilty for millions of records.








