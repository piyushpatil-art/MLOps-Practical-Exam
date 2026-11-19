"""Student 3: create and query a local Feast feature store."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, Int64

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "feature_repo"
FEATURE_FILE = ROOT / "data" / "processed" / "wine_features.parquet"

FEATURE_COLUMNS = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline"
]

def definitions():
    wine = Entity(name="wine", join_keys=["wine_id"], description="One row per wine sample")
    source = FileSource(
        name="wine_features_source",
        path=str(FEATURE_FILE),
        timestamp_field="event_timestamp",
    )
    view = FeatureView(
        name="wine_features",
        entities=[wine],
        ttl=pd.Timedelta(days=3650).to_pytimedelta(),
        schema=[
            Field(name=column, dtype=Float32) for column in FEATURE_COLUMNS
        ],
        source=source,
        online=True,
        tags={"dataset": "sklearn_wine"},
    )
    return wine, view

def get_store():
    if not (REPO / "feature_store.yaml").exists():
        raise FileNotFoundError("feature_repo/feature_store.yaml is missing.")
    return FeatureStore(repo_path=str(REPO))

def apply():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError("Run python src/preprocess.py first.")
    wine, view = definitions()
    get_store().apply([wine, view])
    print("Feast entity and wine_features view registered.")

def materialize():
    store = get_store()
    now = datetime.now(timezone.utc)
    store.materialize(start_date=now - pd.Timedelta(days=3650).to_pytimedelta(), end_date=now)
    print("Feast online store materialized.")

def check():
    store = get_store()
    ids = pd.read_csv(ROOT / "data" / "processed" / "train.csv", usecols=["wine_id"])
    entity_df = ids.head(5).copy()
    entity_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    feature_refs = [f"wine_features:{name}" for name in FEATURE_COLUMNS]
    result = store.get_historical_features(
        entity_df=entity_df, features=feature_refs
    ).to_df()
    print(result.head())
    print(f"Feature retrieval successful: {len(result)} rows")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["apply", "materialize", "check"])
    action = parser.parse_args().action
    {"apply": apply, "materialize": materialize, "check": check}[action]()

if __name__ == "__main__":
    main()
