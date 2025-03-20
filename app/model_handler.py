from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from scipy.stats import uniform, randint
import shap
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelHandler:
    def __init__(self, x, y, preprocessor, feature_names):
        self.x = x
        self.y = y
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.model = None
        self.shap_values = None

    def tune_and_evaluate_models(self):
        logger.info("MODEL SELECTION, TRAINING AND TUNING...")
        x_temp, x_test, y_temp, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        logger.info("RandomizedSearchCV used for tuning, 60/20/20 train/validation/test split")
        
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', 
                            n_jobs=-1, scale_pos_weight=scale_pos_weight)
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

        xgb_params = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5)
        }

        rf_params = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2']
        }

        xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=50, cv=5, n_jobs=-1, scoring='roc_auc', random_state=42)
        rf_search = RandomizedSearchCV(rf, rf_params, n_iter=50, cv=5, n_jobs=-1, scoring='roc_auc', random_state=42)

        xgb_search.fit(x_train, y_train)
        rf_search.fit(x_train, y_train)

        best_xgb = xgb_search.best_estimator_
        best_rf = rf_search.best_estimator_

        xgb_roc_auc = roc_auc_score(y_val, best_xgb.predict_proba(x_val)[:, 1])
        rf_roc_auc = roc_auc_score(y_val, best_rf.predict_proba(x_val)[:, 1])

        self.model = best_xgb if xgb_roc_auc > rf_roc_auc else best_rf
        logger.info(f"SELECTED {'XGBoost' if xgb_roc_auc > rf_roc_auc else 'RandomForest'} model with ROC-AUC: {max(xgb_roc_auc, rf_roc_auc):.4f}")
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        y_prob = self.model.predict_proba(x_test)[:, 1]
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }

        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        logger.info(f"MODEL TRAINING COMPLETED. Final Test Set Metrics: {metrics} \n Metric Justification: - ROC-AUC: Chosen for balanced performance across classes, reflecting overall ranking ability.")
        return self.model, metrics

    def save_model(self, folder="models"):
        logger.info(f"Saving model to {folder}/...")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "model.pkl"), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(folder, "preprocessor.pkl"), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        logger.info("MODEL AND PREPROCESSOR SAVED SUCCESFULLY.")

    def predict(self, x_new):
        x_processed = self.preprocessor.transform(x_new)
        predictions = self.model.predict(x_processed)
        probabilities = self.model.predict_proba(x_processed)[:, 1]
        return predictions, probabilities

    def get_shap_values(self, x_new):
        x_processed = self.preprocessor.transform(x_new)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(x_processed)
        return shap_values

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)