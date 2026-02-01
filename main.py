from src.data.preprocess import load_data, clean_data, preprocess_features, handle_imbalance, train_val_test_split
from src.models.train import train_models
from src.models.evaluate import compare_models, evaluate_model
from src.utils.logger import get_logger
import joblib
import os

logger = get_logger("MainPipeline")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # --- Load & Clean ---
    df = load_data("data/raw/diabetes.csv")
    df = clean_data(df)

    # --- Preprocessing ---
    df = preprocess_features(df)

    # --- Split X/y ---
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    # --- Handle class imbalance ---
    X_res, y_res = handle_imbalance(X, y)

    # --- Train/Val/Test split & save ---
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_res, y_res)

    # --- Train Models ---
    models = train_models(X_train, y_train)

    # --- Validate and Compare Models ---
    results_val = compare_models(models, X_val, y_val)
    logger.info("Validation Results: {}".format(results_val))

    # --- Select best model (F1 score) ---
    best_model_name = max(results_val, key=lambda k: results_val[k]["F1"])
    best_model = models[best_model_name]
    best_model_path = os.path.join(MODEL_DIR,f"{best_model_name}_Best.pkl")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Best model {best_model_name} saved at {best_model_path}")

    # --- Evaluate on Test Set ---
    results_test = evaluate_model(best_model, X_test, y_test, model_name=f"{best_model_name}_Test")
    logger.info(f"Test Results: {results_test}")

if __name__ == "__main__":
    main()
