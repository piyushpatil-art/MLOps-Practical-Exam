"""Student 2: validate and split the Wine dataset reproducibly."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wine.csv"
PROCESSED = ROOT / "data" / "processed"
TARGET = "target"
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline"
]

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. Run: python src/load_data.py")

    df = pd.read_csv(RAW)
    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df[required].isnull().any().any():
        raise ValueError("Dataset contains missing values; inspect the source data.")

    # Stable row IDs allow Feast features to be joined back to their labels.
    df = df[required].copy()
    df.insert(0, "wine_id", range(1, len(df) + 1))

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET]
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED / "train.csv", index=False)
    test_df.to_csv(PROCESSED / "test.csv", index=False)

    # Feast's file source requires an event timestamp column.
    feature_df = df[["wine_id"] + FEATURES].copy()
    feature_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    feature_df.to_parquet(PROCESSED / "wine_features.parquet", index=False)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Processed files written to {PROCESSED}")

if __name__ == "__main__":
    main()
