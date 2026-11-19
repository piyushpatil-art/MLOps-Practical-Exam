"""Student 1: export the Scikit-learn Wine dataset as a CSV."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "wine.csv"

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_wine(as_frame=True)
    df = dataset.frame.copy()
    # Keep the target as an integer class label (0, 1, or 2).
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns)} columns to {OUTPUT}")
    print(df.head())

if __name__ == "__main__":
    main()
