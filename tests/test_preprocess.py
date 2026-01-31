# tests/test_preprocess.py
import pytest
import pandas as pd
from src.data.preprocess import clean_data, preprocess_features, handle_imbalance

def test_preprocessing_pipeline():
    # --- Create a small dummy dataframe with enough samples per class for SMOTE ---
    data = {
        "Pregnancies": [1, 0, 3, 2, 4, 5, 0, 2, 3, 1, 4, 2],
        "Glucose": [85, 0, 120, 90, 100, 95, 88, 110, 105, 99, 115, 92],
        "BloodPressure": [70, 0, 80, 72, 78, 75, 70, 80, 76, 74, 79, 73],
        "SkinThickness": [20, 0, 30, 25, 28, 27, 22, 30, 26, 24, 29, 23],
        "Insulin": [0, 0, 80, 70, 65, 72, 68, 85, 77, 66, 80, 71],
        "BMI": [18.0, 0.0, 28.0, 25.0, 30.0, 27.0, 22.0, 29.0, 26.0, 24.0, 28.5, 23.0],
        "DiabetesPedigreeFunction": [0.5, 0.3, 0.8, 0.4, 0.6, 0.7, 0.3, 0.5, 0.6, 0.4, 0.7, 0.3],
        "Age": [25, 35, 45, 30, 40, 50, 28, 38, 48, 33, 43, 53],
        "Outcome": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

    }
    df = pd.DataFrame(data)

    # --- Clean the data ---
    df_clean = clean_data(df)
    numeric_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    assert not df_clean[numeric_cols].isna().any().any()

    # --- Feature engineering + preprocessing ---
    df_processed = preprocess_features(df_clean)

    # Check categorical columns are gone and replaced with dummies
    for cat in ['AgeGroup', 'BMI_Category', 'BMI_Risk']:
        assert cat not in df_processed.columns

    # Check numeric columns are scaled (mean ~0), exclude Outcome
    numeric_cols_after = df_processed.select_dtypes(include=['float64', 'int64']).columns.drop('Outcome')
    means = df_processed[numeric_cols_after].mean().abs()
    assert (means < 1e-6).all()

    # --- Handle imbalance ---
    X = df_processed.drop(columns=['Outcome'])
    y = df_processed['Outcome']

    # Use k_neighbors=1 for small test dataset
    from imblearn.over_sampling import SMOTE
    X_res, y_res = SMOTE(random_state=42, k_neighbors=1).fit_resample(X, y)

    # Check the resampled data has more samples than original
    assert X_res.shape[0] > X.shape[0]
    assert set(y_res.unique()) == {0, 1}

    # Optional: check that features are still numeric after resampling
    assert all([pd.api.types.is_numeric_dtype(X_res[col]) for col in X_res.columns])
