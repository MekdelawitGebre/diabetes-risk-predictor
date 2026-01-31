from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.logger import get_logger
import joblib
import os

logger = get_logger("ModelTraining")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(X_train, y_train):
    # --- Logistic Regression ---
    logger.info("Training Logistic Regression")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    # --- Random Forest ---
    logger.info("Training Random Forest with GridSearchCV")
    param_grid = {'n_estimators':[100,200],'max_depth':[3,5,7]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    rf.fit(X_train, y_train)
    best_rf = rf.best_estimator_
    logger.info(f"Best RF params: {rf.best_params_}")

    # --- XGBoost ---
    logger.info("Training XGBoost")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)

    # Save individual models
    joblib.dump(lr, os.path.join(MODEL_DIR,"LogisticRegression.pkl"))
    joblib.dump(best_rf, os.path.join(MODEL_DIR,"RandomForest.pkl"))
    joblib.dump(xgb, os.path.join(MODEL_DIR,"XGBoost.pkl"))

    return {"LogisticRegression": lr, "RandomForest": best_rf, "XGBoost": xgb}