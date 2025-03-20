from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from .data_handler import DataHandler
from .model_handler import ModelHandler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Initializing FastAPI application..")
app = FastAPI(title="Peswa Loan Default Predictor")

logger.info("Setting up data handler and training model...")
data_handler = DataHandler()
df = data_handler.load_and_describe()
df_cleaned = data_handler.clean_data(df)
x, y, feature_names = data_handler.preprocess(df_cleaned)
model_handler = ModelHandler(x, y, data_handler.preprocessor, feature_names)

model, metrics = model_handler.tune_and_evaluate_models()
model_handler.save_model()

class LoanInput(BaseModel):
    loan_amount: float
    term: int
    interest_rate: float
    monthly_income: float
    credit_score: int
    employment_status: str
    loan_purpose: str
    num_previous_loans: int
    default_history: int
    debt_to_income_ratio: float = None  

@app.post("/predict")
def predict_loan_default(loan_input: LoanInput): 
    logger.info("Processing /predict request...")
    input_data = pd.DataFrame([loan_input.model_dump()])
    input_data = data_handler.feature_engineering(input_data)
    predictions, probabilities = model_handler.predict(input_data)
    result = {"default": bool(predictions[0]), "probability": float(probabilities[0])}
    logger.info(f"/predict completed: {result}")
    return result
  
@app.post("/risk_level")
def get_risk_level(loan_input: LoanInput):
    logger.info("Processing /risk_level request...")
    input_data = pd.DataFrame([loan_input.model_dump()])
    input_data = data_handler.feature_engineering(input_data)
    _, probability = model_handler.predict(input_data)
    prob = float(probability[0])
    credit_score = input_data['credit_score'].iloc[0]
    dti = input_data['debt_to_income_ratio'].iloc[0]

    if prob < 0.20:
        risk = "Low"
        factor = "Low debt-to-income" if dti < 0.3 else "Good credit score" if credit_score >= 670 else "Stable factors"
    elif 0.20 <= prob <= 0.50:
        risk = "Medium"
        factor = "Moderate debt-to-income" if 0.3 <= dti <= 0.5 else "Fair credit score" if 580 <= credit_score < 670 else "Mixed factors"
    else:
        risk = "High"
        factor = "High debt-to-income" if dti > 0.5 else "Poor credit score" if credit_score < 580 else "High risk history"
    
    result = {"risk_level": risk, "key_factor": factor, "probability": prob}
    logger.info(f"/risk_level completed: {result}")
    return result
    
@app.post("/interpretability")
def get_individual_interpretability(loan_input: LoanInput):
    logger.info("Processing /interpretability request...")
    input_data = pd.DataFrame([loan_input.model_dump()])
    input_data = data_handler.feature_engineering(input_data)
    shap_values = model_handler.get_shap_values(input_data)
    shap_dict = dict(zip(model_handler.feature_names, shap_values[0].tolist()))
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    interpretation = {
        "top_factors": [{"feature": f, "impact": float(round(i, 4))} for f, i in top_factors], 
        "insight": ("High default risk is likely driven by poor credit score or high debt-to-income ratio" 
                    if any("credit_score" in f or "debt_to_income" in f for f, _ in top_factors) else 
                    "Risk influenced by loan-specific factors like amount or term.")
    }
    logger.info(f"/interpretability completed: Top factors - {[(f, round(i, 4)) for f, i in top_factors]}")
    return interpretation

@app.get("/feature_importance")
def get_feature_importance():
    logger.info("Processing /feature_importance request...")
    importance = model_handler.get_feature_importance()
    feature_importance = [{"feature": feature, "importance": float(imp.item())} for feature, imp in importance]
    logger.info(f"/feature_importance completed: Top 3 - {feature_importance[:3]}")
    return {"feature_importance": feature_importance}