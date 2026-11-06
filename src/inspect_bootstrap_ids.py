import pandas as pd

path = "outputs/feature_fusion/predictions.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print("\nUnique experiments:")
print(df["Experiment"].unique())

print("\nFirst 20 rows:")
print(df[[
    "Experiment", "VideoID", "Label",
    "Actual", "Prediction"
]].head(20).to_string(index=False))

print("\nVideoID data type:", df["VideoID"].dtype)
print("\nNumber of unique VideoIDs:", df["VideoID"].nunique())

# Inspect one experiment only
model = df["Experiment"].iloc[0]
one = df[df["Experiment"] == model].copy()

print("\nFirst experiment:", model)
print("Rows:", len(one))

print("\nActual label counts:")
print(one["Actual"].value_counts(dropna=False))

print("\nNumber of videos per VideoID:")
print(one.groupby("VideoID").size().value_counts().sort_index())

print("\nFirst 15 VideoID groups:")
print(
    one.groupby("VideoID")["Actual"]
       .agg(["count", "unique"])
       .head(15)
       .to_string()
)

print("\nCheck for IDs with both labels:")
valid_pairs = 0
for video_id, group in one.groupby("VideoID"):
    labels = set(group["Actual"].astype(int))
    if labels == {0, 1}:
        valid_pairs += 1

print("IDs containing both real and fake:", valid_pairs)
