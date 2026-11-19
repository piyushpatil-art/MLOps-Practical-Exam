
"""Member 6: MLflow experiment tracking and best-model integration."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from model_training import train_candidates


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
METRICS_FILE = ROOT / "metrics.json"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("Wine_Classification")

    results = train_candidates()

    best_result = max(results, key=lambda item: item["accuracy"])

    metrics_summary = {}

    for result in results:
        name = result["name"]
        model = result["model"]
        X_test = result["X_test"]
        y_test = result["y_test"]

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("features", len(X_test.columns))
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )


        metrics_summary[name] = metrics

        print(f"\n{name}: {metrics}")

    best_model = best_result["model"]
    best_name = best_result["name"]

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    with open(MODEL_DIR / "best_model.txt", "w") as file:
        file.write(f"Best Model: {best_name}\n")
        file.write(f"Accuracy: {best_result['accuracy']:.4f}\n")

    with open(METRICS_FILE, "w") as file:
        json.dump(
            {
                "best_model": best_name,
                "best_accuracy": best_result["accuracy"],
                "all_models": metrics_summary,
            },
            file,
            indent=4,
        )

    print("\nBest model:", best_name)
    print("Best accuracy:", best_result["accuracy"])
    print("Model and metrics saved successfully.")


if __name__ == "__main__":
    main()
