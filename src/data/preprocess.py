import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger
from src.data.feature_engineering import add_age_bins, add_bmi_features

logger = get_logger("Preprocessing")

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data(path="../data/raw/diabetes.csv") -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data: replacing zeros with median and removing duplicates")
    cols_invalid_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[cols_invalid_zero] = df[cols_invalid_zero].replace(0, np.nan)
    for col in cols_invalid_zero:
        df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    return df

def preprocess_features(df):
    # --- Feature Engineering ---
    df = add_age_bins(df)
    df = add_bmi_features(df)

    # --- Encode categorical variables ---
    categorical_cols = ['AgeGroup', 'BMI_Category', 'BMI_Risk']
    # Only encode if these columns exist (safe for single row input)
    existing_cats = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # --- Scaling numeric features (exclude Outcome if it exists) ---
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    if 'Outcome' in numerical_cols:
        numerical_cols = numerical_cols.drop('Outcome')
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df



def handle_imbalance(X: pd.DataFrame, y: pd.Series):
    logger.info("Handling class imbalance with SMOTE")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_val_test_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, val_size=0.1, seed=42):
    logger.info("Splitting data into train/validation/test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=seed
    )

    # --- Save splits ---
    X_train.to_csv(os.path.join(PROCESSED_DIR,"X_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_DIR,"X_val.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR,"y_train.csv"), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR,"y_val.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR,"y_test.csv"), index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test
