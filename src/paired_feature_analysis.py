import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA_PATH = "outputs/features/uadfv_features.csv"
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_columns = [
    c for c in df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

real = df[df["label"] == "real"].set_index("video_id")
fake = df[df["label"] == "fake"].set_index("video_id")

common_ids = sorted(set(real.index) & set(fake.index))

if len(common_ids) != 49:
    raise ValueError(f"Expected 49 matched pairs, found {len(common_ids)}")

results = []

for feature in feature_columns:
    real_values = real.loc[common_ids, feature].astype(float).to_numpy()
    fake_values = fake.loc[common_ids, feature].astype(float).to_numpy()

    differences = fake_values - real_values

    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        try:
            p_value = float(
                wilcoxon(
                    fake_values,
                    real_values,
                    alternative="two-sided",
                    zero_method="wilcox"
                ).pvalue
            )
        except ValueError:
            p_value = 1.0

    results.append({
        "feature": feature,
        "feature_type": (
            "Eigenvalue" if feature.startswith("eigen_")
            else "Eigen-jitter"
        ),
        "real_median": float(np.median(real_values)),
        "fake_median": float(np.median(fake_values)),
        "median_fake_minus_real": float(np.median(differences)),
        "pairs_fake_higher": int(np.sum(differences > 0)),
        "pairs_real_higher": int(np.sum(differences < 0)),
        "p_value": p_value
    })

result_df = pd.DataFrame(results)

# Benjamini-Hochberg false-discovery-rate correction.
p_values = result_df["p_value"].to_numpy()
m = len(p_values)
order = np.argsort(p_values)
ranked = p_values[order]

adjusted = np.empty(m)
adjusted_sorted = ranked * m / np.arange(1, m + 1)
adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
adjusted[order] = np.minimum(adjusted_sorted, 1.0)

result_df["adjusted_p_value"] = adjusted

result_df = result_df.sort_values(
    ["adjusted_p_value", "p_value"],
    ascending=True
)

output_path = os.path.join(
    OUTPUT_DIR, "paired_feature_statistics.csv"
)
result_df.to_csv(output_path, index=False)

print("=" * 70)
print("PAIRED REAL VS FAKE FEATURE ANALYSIS")
print("=" * 70)
print("Matched pairs:", len(common_ids))
print("Features tested:", len(feature_columns))
print("Features significant after correction (q < 0.05):",
      int((result_df["adjusted_p_value"] < 0.05).sum()))

print("\nTOP 15 FEATURES")
print(result_df.head(15).to_string(index=False))

print("\nTOP 10 EIGEN-JITTER FEATURES")
jitter_results = result_df[
    result_df["feature_type"] == "Eigen-jitter"
]
print(jitter_results.head(10).to_string(index=False))

print("\nSaved:", output_path)
print("\nAnalysis complete.")
