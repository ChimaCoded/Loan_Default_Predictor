import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataHandler:
    def __init__(self):
        self.numerical_cols = ['loan_amount', 'term', 'interest_rate', 'monthly_income', 'credit_score', 
                               'num_previous_loans', 'default_history', 'debt_to_income_ratio', 'monthly_payment', 
                               'total_loan_burden']
        self.categorical_cols = ['employment_status', 'loan_purpose']
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self):
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        return ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

    def load_and_describe(self, filepath="data/loan_data.csv"):
        logger.info(f"Loading loan dataset from {filepath}...")
        df = pd.read_csv(filepath)
        logger.info(f"DATASET LOADED SUCCESFULLY. Shape: {df.shape}, Default Rate: {df['loan_default'].mean():.4f}")
        logger.info(f"Missing Values:\n{df.isnull().sum()}")
        logger.info(f"Basic Statistics:\n{df.describe()}")
        return df

    def clean_data(self, df):
        logger.info("CLEANING DATASET...")
        df = df.dropna()
        df['loan_default'] = df['loan_default'].astype(int)
        for col in ['loan_amount', 'interest_rate', 'monthly_income']:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
        df = df[(df['loan_amount'] > 0) & (df['term'] > 0) & (df['monthly_income'] >= 0) & 
                (df['interest_rate'] >= 0) & (df['credit_score'] >= 0)]
        logger.info(f"DATASET CLEANED SUCCESFULLY. Shape: {df.shape}")
        return df

    def feature_engineering(self, df):
        monthly_rate = df['interest_rate'] / 1200
        df['monthly_payment'] = (df['loan_amount'] * monthly_rate * (1 + monthly_rate) ** df['term']) / \
                                ((1 + monthly_rate) ** df['term'] - 1)
        df['monthly_payment'] = df['monthly_payment'].replace([np.inf, -np.inf], np.nan).fillna(df['loan_amount'] / df['term'])
        df['monthly_payment'] = df['monthly_payment'].clip(upper=df['loan_amount'])
        df['debt_to_income_ratio'] = df['monthly_payment'] / df['monthly_income'].replace(0, 1)
        df['debt_to_income_ratio'] = df['debt_to_income_ratio'].replace([np.inf, -np.inf], 10).clip(upper=10)
        df['total_loan_burden'] = df['loan_amount'] * (df['num_previous_loans'] + 1)
        return df

    def preprocess(self, df):
        logger.info("Preprocessing data with feature engineering..")
        df = self.feature_engineering(df)
        x = df[self.numerical_cols + self.categorical_cols]
        y = df['loan_default'] if 'loan_default' in df.columns else None
        x_processed = self.preprocessor.fit_transform(x) if y is not None else self.preprocessor.transform(x)
        logger.info(f"DATA PREPROCESSED SUCCESFULLY. Shape: {x_processed.shape}")
        return x_processed, y, self.numerical_cols + self.categorical_cols