import os
import numpy as np
import pandas as pd

INPUT_CSV = "outputs/spectral/spectral_features.csv"
OUTPUT_DIR = "outputs/features"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "sample_video_features.csv"
)


def extract_video_features(df):
    features = {}

    features["frame_count"] = len(df)

    # Eigenvalue 1 is excluded because it is theoretically zero.
    for i in range(2, 21):

        eigen_column = f"eigen_{i}"
        jitter_column = f"jitter_eigen_{i}"

        eigen = df[eigen_column].dropna().to_numpy()
        jitter = df[jitter_column].dropna().to_numpy()

        features[f"eigen_{i}_mean"] = np.mean(eigen)
        features[f"eigen_{i}_std"] = np.std(eigen)
        features[f"eigen_{i}_min"] = np.min(eigen)
        features[f"eigen_{i}_max"] = np.max(eigen)

        features[f"jitter_{i}_mean"] = np.mean(jitter)
        features[f"jitter_{i}_std"] = np.std(jitter)
        features[f"jitter_{i}_mean_abs"] = np.mean(np.abs(jitter))
        features[f"jitter_{i}_max_abs"] = np.max(np.abs(jitter))

    # Overall temporal variation across the selected eigenvalues.
    jitter_matrix = df[
        [f"jitter_eigen_{i}" for i in range(2, 21)]
    ].to_numpy(dtype=float)

    features["overall_mean_absolute_jitter"] = np.nanmean(
        np.abs(jitter_matrix)
    )

    features["overall_jitter_std"] = np.nanstd(jitter_matrix)

    return features


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading spectral feature data...")

    df = pd.read_csv(INPUT_CSV)

    if len(df) < 3:
        raise ValueError("Not enough frames to extract video features.")

    print(f"Frames found: {len(df)}")

    print("\nExtracting video-level statistical features...")

    features = extract_video_features(df)

    result = pd.DataFrame([features])

    result.insert(0, "video_name", "real_sample.mp4")
    result.insert(1, "label", "unlabeled")

    result.to_csv(OUTPUT_CSV, index=False)

    print("\n========== PHASE 5 COMPLETE ==========")
    print(f"Video feature rows: {len(result)}")
    print(f"Total feature columns: {len(result.columns) - 2}")
    print(f"Output: {OUTPUT_CSV}")

    print("\nFeature preview:")
    print(result.iloc[:, :12].to_string(index=False))

    print("\nLabel is currently UNLABELED.")
    print("Do not use this row for supervised training yet.")


if __name__ == "__main__":
    main()
