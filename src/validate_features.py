import os
import numpy as np
import pandas as pd

path = "outputs/features/uadfv_features.csv"
df = pd.read_csv(path)

print("=" * 60)
print("FEATURE DATASET VALIDATION")
print("=" * 60)

print("\n1. Dataset dimensions")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. Class distribution")
print(df["label"].value_counts().to_string())

print("\n3. Pair verification")
pair_counts = df.groupby("video_id")["label"].nunique()
print("Unique video IDs:", df["video_id"].nunique())
print("IDs containing both real and fake:", (pair_counts == 2).sum())
print("IDs with missing class:", (pair_counts != 2).sum())

print("\n4. Missing values")
print("Total missing cells:", int(df.isna().sum().sum()))

numeric = df.select_dtypes(include=[np.number])
print("\n5. Non-finite values")
print("Non-finite cells:", int((~np.isfinite(numeric)).sum().sum()))

feature_columns = [
    c for c in numeric.columns
    if c.startswith("eigen_") or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

features = df[feature_columns]
print("\n6. Spectral feature checks")
print("Spectral feature count:", len(feature_columns))
print("Constant features:", int((features.nunique() <= 1).sum()))

print("\n7. Feature statistics by class")
for label in ["real", "fake"]:
    subset = df[df["label"] == label]
    print(
        f"{label.upper()}: "
        f"mean absolute jitter = "
        f"{subset['overall_jitter_mean_abs'].mean():.8f}"
    )

print("\n8. Sample rows")
print(df[["video_id", "video_name", "label",
          "frame_count", "overall_jitter_mean_abs"]].head(10).to_string(index=False))

print("\n" + "=" * 60)
if (
    len(df) == 98
    and df["label"].value_counts().get("real", 0) == 49
    and df["label"].value_counts().get("fake", 0) == 49
    and df["video_id"].nunique() == 49
    and (pair_counts == 2).all()
    and df.isna().sum().sum() == 0
    and np.isfinite(numeric.to_numpy()).all()
):
    print("VALIDATION PASSED")
else:
    print("VALIDATION NEEDS ATTENTION")
print("=" * 60)
