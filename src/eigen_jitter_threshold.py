import os
import warnings
import numpy as np
import pandas as pd

from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")

DATA_PATH = "outputs/features/uadfv_features.csv"
CACHE_DIR = "outputs/fixed_sigma/aligned_landmarks"
OUT_DIR = "outputs/eigen_jitter_threshold"

os.makedirs(OUT_DIR, exist_ok=True)

NODE_COUNT = 468
EIGEN_COUNT = 20
FOLDS = 5

print("=" * 65)
print("EXACT EIGEN-JITTER THRESHOLD EXPERIMENT")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["video_id"] = df["video_id"].astype(str)

df = df.sort_values(
    ["video_id", "label"]
).reset_index(drop=True)

y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].to_numpy()

if len(df) != 98 or len(np.unique(groups)) != 49:
    raise ValueError("Expected 98 videos and 49 matched pairs.")

edges = np.array(sorted([
    tuple(sorted(edge))
    for edge in __import__("mediapipe").solutions.face_mesh.FACEMESH_TESSELATION
]), dtype=np.int32)
edges = np.unique(edges, axis=0)

print("Videos:", len(df))
print("Matched pairs:", len(np.unique(groups)))
print("Graph edges:", len(edges))

# Load cached aligned landmarks.
cache = {}

for _, row in df.iterrows():
    key = (row["video_id"], row["label"])
    path = os.path.join(
        CACHE_DIR, f"{row['label']}_{row['video_id']}.npy"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    cache[key] = np.load(path).astype(np.float64)

print("Loaded all landmark files.")

def get_jitter_signal(aligned, sigma):
    distances = np.linalg.norm(
        aligned[:, edges[:, 0], :]
        - aligned[:, edges[:, 1], :],
        axis=2
    )

    eigenvalue_frames = []

    for frame_distances in distances:
        weights = np.exp(
            -(frame_distances ** 2) / (2.0 * sigma ** 2)
        )

        adjacency = np.zeros(
            (NODE_COUNT, NODE_COUNT), dtype=np.float64
        )

        adjacency[edges[:, 0], edges[:, 1]] = weights
        adjacency[edges[:, 1], edges[:, 0]] = weights

        degree = adjacency.sum(axis=1)

        if np.any(degree <= 1e-12):
            raise ValueError("Graph contains an isolated node.")

        inv = 1.0 / np.sqrt(degree)

        normalized_adjacency = (
            inv[:, None] * adjacency * inv[None, :]
        )

        laplacian = np.eye(NODE_COUNT) - normalized_adjacency

        eigenvalues = eigh(
            laplacian,
            subset_by_index=[0, EIGEN_COUNT - 1],
            eigvals_only=True,
            check_finite=False,
            driver="evr"
        )

        eigenvalue_frames.append(eigenvalues)

    eigenvalues = np.asarray(eigenvalue_frames)

    # Exact vector second-order difference from the PPT.
    acceleration = (
        eigenvalues[2:]
        - 2.0 * eigenvalues[1:-1]
        + eigenvalues[:-2]
    )

    # L2 norm across the 20 eigenvalues at each time step.
    jitter = np.linalg.norm(acceleration, axis=1)

    return jitter

# Calculate one video-level score: mean exact J(t).
# Sigma is the same per-video median edge-distance method
# used in the original feature extraction.
scores = []

for i, row in df.iterrows():
    key = (row["video_id"], row["label"])
    aligned = cache[key]

    distances = np.linalg.norm(
        aligned[:, edges[:, 0], :]
        - aligned[:, edges[:, 1], :],
        axis=2
    )

    sigma = float(np.median(distances))

    if sigma <= 0:
        raise ValueError(f"Invalid sigma for {key}")

    jitter = get_jitter_signal(aligned, sigma)

    scores.append({
        "video_id": row["video_id"],
        "label": row["label"],
        "actual": int(y[i]),
        "sigma": sigma,
        "frame_count": len(aligned),
        "jitter_mean": float(np.mean(jitter)),
        "jitter_median": float(np.median(jitter)),
        "jitter_std": float(np.std(jitter)),
        "jitter_p95": float(np.percentile(jitter, 95)),
        "jitter_max": float(np.max(jitter))
    })

    if (i + 1) % 10 == 0 or i == len(df) - 1:
        print(f"Processed {i + 1}/{len(df)} videos")

scores_df = pd.DataFrame(scores)
scores_df.to_csv(
    os.path.join(OUT_DIR, "video_jitter_scores.csv"),
    index=False
)

print("\nJitter score summary:")
print(
    scores_df.groupby("label")["jitter_mean"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_string()
)

# Pair-aware five-fold threshold evaluation.
# Threshold and direction are selected using training videos only.
cv = GroupKFold(n_splits=FOLDS)

all_predictions = np.full(len(df), -1, dtype=int)
all_scores = scores_df["jitter_mean"].to_numpy()
fold_results = []

def select_threshold(train_values, train_labels):
    unique = np.unique(train_values)

    if len(unique) == 1:
        candidates = np.array([unique[0]])
    else:
        midpoints = (
            unique[:-1] + unique[1:]
        ) / 2.0
        candidates = np.concatenate([
            [unique[0] - 1e-12],
            midpoints,
            [unique[-1] + 1e-12]
        ])

    best = None

    for direction in ["higher_is_fake", "lower_is_fake"]:
        for threshold in candidates:
            if direction == "higher_is_fake":
                pred = (train_values >= threshold).astype(int)
            else:
                pred = (train_values <= threshold).astype(int)

            score = balanced_accuracy_score(train_labels, pred)

            if best is None or score > best["balanced_accuracy"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "balanced_accuracy": float(score)
                }

    return best

for fold, (train_idx, test_idx) in enumerate(
    cv.split(df, y, groups), start=1
):
    train_ids = set(groups[train_idx])
    test_ids = set(groups[test_idx])

    if train_ids & test_ids:
        raise RuntimeError("Pair leakage detected.")

    train_values = all_scores[train_idx]
    test_values = all_scores[test_idx]

    chosen = select_threshold(train_values, y[train_idx])

    if chosen["direction"] == "higher_is_fake":
        predictions = (
            test_values >= chosen["threshold"]
        ).astype(int)
    else:
        predictions = (
            test_values <= chosen["threshold"]
        ).astype(int)

    all_predictions[test_idx] = predictions

    fold_acc = accuracy_score(y[test_idx], predictions)

    fold_results.append({
        "fold": fold,
        "training_pairs": len(train_ids),
        "testing_pairs": len(test_ids),
        "threshold": chosen["threshold"],
        "direction": chosen["direction"],
        "training_balanced_accuracy": chosen["balanced_accuracy"],
        "test_accuracy": fold_acc,
        "test_balanced_accuracy": balanced_accuracy_score(
            y[test_idx], predictions
        )
    })

    print(
        f"\nFold {fold}: threshold={chosen['threshold']:.10f}, "
        f"direction={chosen['direction']}, "
        f"test accuracy={fold_acc:.4f}"
    )

print("\n" + "=" * 65)
print("OUT-OF-FOLD THRESHOLD RESULTS")
print("=" * 65)

print("Accuracy:", accuracy_score(y, all_predictions))
print("Precision:", precision_score(y, all_predictions, zero_division=0))
print("Recall:", recall_score(y, all_predictions, zero_division=0))
print("F1:", f1_score(y, all_predictions, zero_division=0))
print("Balanced accuracy:", balanced_accuracy_score(y, all_predictions))
print("ROC-AUC of raw jitter score:", roc_auc_score(y, all_scores))

print("\nConfusion matrix:")
print(confusion_matrix(y, all_predictions))

fold_df = pd.DataFrame(fold_results)
print("\nFold results:")
print(fold_df.to_string(index=False))

fold_df.to_csv(
    os.path.join(OUT_DIR, "fold_results.csv"),
    index=False
)

prediction_df = scores_df.copy()
prediction_df["predicted"] = all_predictions
prediction_df["prediction_label"] = np.where(
    all_predictions == 1, "fake", "real"
)

prediction_df.to_csv(
    os.path.join(OUT_DIR, "predictions.csv"),
    index=False
)

summary = pd.DataFrame([{
    "accuracy": accuracy_score(y, all_predictions),
    "precision": precision_score(y, all_predictions, zero_division=0),
    "recall": recall_score(y, all_predictions, zero_division=0),
    "f1": f1_score(y, all_predictions, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y, all_predictions),
    "roc_auc_raw_score": roc_auc_score(y, all_scores)
}])

summary.to_csv(
    os.path.join(OUT_DIR, "summary.csv"),
    index=False
)

print("\nSaved results to:", OUT_DIR)
print("EXPERIMENT COMPLETE")
