import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("outputs/features/uadfv_features.csv")

real = df[df["label"] == "real"].set_index("video_id")
fake = df[df["label"] == "fake"].set_index("video_id")

common_ids = sorted(set(real.index) & set(fake.index))

print("=" * 65)
print("DATASET CONFOUNDING AUDIT")
print("=" * 65)

results = []

for feature in ["frame_count", "sigma"]:
    real_values = real.loc[common_ids, feature].astype(float).to_numpy()
    fake_values = fake.loc[common_ids, feature].astype(float).to_numpy()

    differences = fake_values - real_values

    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(fake_values, real_values).pvalue
        )

    results.append({
        "variable": feature,
        "real_median": np.median(real_values),
        "fake_median": np.median(fake_values),
        "median_fake_minus_real": np.median(differences),
        "fake_higher_pairs": int(np.sum(differences > 0)),
        "real_higher_pairs": int(np.sum(differences < 0)),
        "equal_pairs": int(np.sum(differences == 0)),
        "paired_p_value": p_value
    })

result_df = pd.DataFrame(results)

print("\nPaired comparison:")
print(result_df.to_string(index=False))

print("\nCorrelation of metadata with spectral features:")
spectral_columns = [
    c for c in df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

for metadata in ["frame_count", "sigma"]:
    correlations = df[spectral_columns].corrwith(df[metadata]).abs()
    top = correlations.sort_values(ascending=False).head(5)

    print(f"\nTop 5 absolute correlations with {metadata}:")
    print(top.to_string())

output_dir = "outputs/evaluation"
os.makedirs(output_dir, exist_ok=True)
result_df.to_csv(
    os.path.join(output_dir, "metadata_confounding_audit.csv"),
    index=False
)

print("\nSaved: outputs/evaluation/metadata_confounding_audit.csv")
print("\nAudit complete.")
