
import pandas as pd
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float64

# Load Wine dataset
df = pd.read_csv("../data/wine.csv")

# Create wine ID
df["wine_id"] = range(len(df))

# Add timestamp
df["event_timestamp"] = pd.Timestamp.now(tz="UTC")

# Save feature data
df.to_parquet("wine_features.parquet", index=False)

# Create Entity
wine = Entity(
    name="wine",
    join_keys=["wine_id"],
    value_type=Int64
)

# Create data source
wine_source = FileSource(
    name="wine_source",
    path="wine_features.parquet",
    timestamp_field="event_timestamp"
)

# Define features (exclude target and ID)
feature_columns = [
    col for col in df.columns
    if col not in ["wine_id", "target", "event_timestamp"]
]

# Create Feature View
wine_features = FeatureView(
    name="wine_features",
    entities=[wine],
    ttl=timedelta(days=365),
    schema=[
        Field(name=col, dtype=Float64)
        for col in feature_columns
    ],
    source=wine_source
)

print("Wine Feature Store created!")
