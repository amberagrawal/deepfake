import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

folder = Path("outputs/eigen_jitter_threshold")
df = pd.read_csv(folder / "video_jitter_scores.csv")

# 1. Distribution comparison
plt.figure(figsize=(9, 5))

plt.hist(
    df[df["label"] == "real"]["jitter_mean"],
    bins=12, alpha=0.65, label="Real"
)

plt.hist(
    df[df["label"] == "fake"]["jitter_mean"],
    bins=12, alpha=0.65, label="Fake"
)

plt.xlabel("Mean Eigen-Jitter")
plt.ylabel("Number of videos")
plt.title("Eigen-Jitter Distribution: Real vs Fake")
plt.legend()
plt.tight_layout()
plt.savefig(folder / "jitter_distribution.png", dpi=300)
plt.close()

# 2. Matched pair comparison
pivot = df.pivot(
    index="video_id",
    columns="label",
    values="jitter_mean"
)

pivot["difference_fake_minus_real"] = (
    pivot["fake"] - pivot["real"]
)

plt.figure(figsize=(10, 5))

for _, row in pivot.iterrows():
    plt.plot(
        [0, 1],
        [row["real"], row["fake"]],
        marker="o",
        alpha=0.45
    )

plt.xticks([0, 1], ["Real", "Fake"])
plt.ylabel("Mean Eigen-Jitter")
plt.title("Matched Real-Fake Video Pair Comparison")
plt.tight_layout()
plt.savefig(folder / "matched_pair_comparison.png", dpi=300)
plt.close()

# 3. Pair statistics
print("\nMATCHED PAIR ANALYSIS")
print("Total pairs:", len(pivot))
print(
    "Fake jitter lower than real:",
    int((pivot["fake"] < pivot["real"]).sum())
)
print(
    "Fake jitter higher than real:",
    int((pivot["fake"] > pivot["real"]).sum())
)
print(
    "Median paired difference:",
    pivot["difference_fake_minus_real"].median()
)

print("\nSaved:")
print(folder / "jitter_distribution.png")
print(folder / "matched_pair_comparison.png")
