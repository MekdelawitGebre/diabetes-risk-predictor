from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import get_logger

logger = get_logger("Evaluation")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def evaluate_model(model, X, y, model_name="model"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_proba),
        "Confusion Matrix": confusion_matrix(y, y_pred)
    }
    logger.info(f"Evaluation metrics for {model_name}: {metrics}")

    # --- ROC Curve ---
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"{model_name} ROC Curve")
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_roc_curve.png"))
    plt.close()

    # --- Confusion Matrix ---
    cm = metrics["Confusion Matrix"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return metrics

def compare_models(models, X_val, y_val):
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_val, y_val, model_name=name)
    return results