"""Student 4: train and evaluate candidate classifiers using Feast features."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_store import FEATURE_COLUMNS, ROOT, get_store

def get_feature_frame(split_name):
    split_path = ROOT / "data" / "processed" / f"{split_name}.csv"
    labels = pd.read_csv(split_path)
    entity_df = labels[["wine_id"]].copy()
    entity_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    refs = [f"wine_features:{name}" for name in FEATURE_COLUMNS]
    features = get_store().get_historical_features(
        entity_df=entity_df, features=refs
    ).to_df()
    merged = labels[["wine_id", "target"]].merge(
        features.drop(columns=["event_timestamp"], errors="ignore"),
        on="wine_id", how="inner"
    )
    if len(merged) != len(labels):
        raise RuntimeError("Feature Store returned an incomplete set of wine rows.")
    return merged[FEATURE_COLUMNS], merged["target"]

def train_candidates():
    X_train, y_train = get_feature_frame("train")
    X_test, y_test = get_feature_frame("test")

    candidates = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "SVC": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=42)
        ),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n{name} accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))
        results.append({
            "name": name, "model": model, "accuracy": accuracy,
            "X_test": X_test, "y_test": y_test
        })
    return results

if __name__ == "__main__":
    train_candidates()
