import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "outputs/spectral/spectral_features.csv"
OUTPUT_DIR = "outputs/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading spectral features...")

df = pd.read_csv(INPUT_CSV)

eigen_columns = [f"eigen_{i}" for i in range(1, 21)]
jitter_columns = [f"jitter_eigen_{i}" for i in range(1, 21)]

# --------------------------------------------------
# 1. Eigenvalues over time
# --------------------------------------------------

plt.figure(figsize=(14, 7))

for column in eigen_columns:
    plt.plot(df["frame"], df[column], linewidth=1, label=column)

plt.title("Facial Graph Eigenvalues Across Video Frames")
plt.xlabel("Frame Number")
plt.ylabel("Eigenvalue")
plt.grid(alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "eigenvalues_over_time.png"),
    dpi=200
)
plt.close()

print("1/4 Eigenvalue graph saved.")

# --------------------------------------------------
# 2. Eigen-jitter over time
# --------------------------------------------------

plt.figure(figsize=(14, 7))

for column in jitter_columns:
    plt.plot(df["frame"], df[column], linewidth=1, label=column)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Temporal Eigen-Jitter Across Video Frames")
plt.xlabel("Frame Number")
plt.ylabel("Second Difference of Eigenvalues")
plt.grid(alpha=0.3)
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "eigen_jitter_over_time.png"),
    dpi=200
)
plt.close()

print("2/4 Eigen-jitter graph saved.")

# --------------------------------------------------
# 3. Eigenvalue heatmap
# --------------------------------------------------

eigen_matrix = df[eigen_columns].to_numpy().T

plt.figure(figsize=(14, 7))

plt.imshow(
    eigen_matrix,
    aspect="auto",
    interpolation="nearest",
    cmap="viridis"
)

plt.colorbar(label="Eigenvalue")
plt.title("Facial Graph Spectral Heatmap")
plt.xlabel("Frame Position")
plt.ylabel("Eigenvalue Index")

plt.yticks(
    range(20),
    [f"Eigenvalue {i}" for i in range(1, 21)]
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "eigenvalue_heatmap.png"),
    dpi=200
)
plt.close()

print("3/4 Eigenvalue heatmap saved.")

# --------------------------------------------------
# 4. Summary statistics
# --------------------------------------------------

summary = []

for column in jitter_columns:
    values = df[column].dropna()

    summary.append({
        "feature": column,
        "mean": values.mean(),
        "standard_deviation": values.std(),
        "minimum": values.min(),
        "maximum": values.max(),
        "mean_absolute_jitter": values.abs().mean()
    })

summary_df = pd.DataFrame(summary)

summary_path = os.path.join(
    OUTPUT_DIR,
    "jitter_statistics.csv"
)

summary_df.to_csv(summary_path, index=False)

print("4/4 Jitter statistics saved.")

print("\n========== PHASE 4 COMPLETE ==========")
print("\nGenerated files:")

for filename in [
    "eigenvalues_over_time.png",
    "eigen_jitter_over_time.png",
    "eigenvalue_heatmap.png",
    "jitter_statistics.csv"
]:
    print(os.path.join(OUTPUT_DIR, filename))

print("\nJitter statistics preview:")
print(summary_df.head(10).to_string(index=False))
