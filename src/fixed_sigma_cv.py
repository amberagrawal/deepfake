import os
import warnings
import numpy as np
import pandas as pd

from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

DATA_PATH = "outputs/features/uadfv_features.csv"
CACHE_DIR = "outputs/fixed_sigma/aligned_landmarks"
OUT_DIR = "outputs/fixed_sigma_cv"

os.makedirs(OUT_DIR, exist_ok=True)

NODE_COUNT = 468
EIGEN_COUNT = 20
FOLDS = 5

print("=" * 70)
print("FIVE-FOLD PAIR-AWARE SIGMA COMPARISON")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["video_id"] = df["video_id"].astype(str)

df = df.sort_values(
    ["video_id", "label"]
).reset_index(drop=True)

feature_columns = [
    c for c in df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].to_numpy()

if df["video_id"].nunique() != 49 or len(df) != 98:
    raise ValueError("Expected 98 videos arranged in 49 pairs.")

pair_check = df.groupby("video_id")["label"].agg(set)
if not all(v == {"real", "fake"} for v in pair_check):
    raise ValueError("A video ID does not contain both classes.")

edges = np.array(sorted([
    tuple(sorted(edge))
    for edge in __import__("mediapipe").solutions.face_mesh.FACEMESH_TESSELATION
]), dtype=np.int32)
edges = np.unique(edges, axis=0)

print("Videos:", len(df))
print("Pairs:", df["video_id"].nunique())
print("Features:", len(feature_columns))
print("Folds:", FOLDS)
print("Graph edges:", len(edges))

# Load all aligned landmarks once.
aligned_cache = {}

for _, row in df.iterrows():
    video_id = row["video_id"]
    label = row["label"]
    cache_path = os.path.join(
        CACHE_DIR, f"{label}_{video_id}.npy"
    )

    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)

    aligned = np.load(cache_path)

    if aligned.ndim != 3 or aligned.shape[1:] != (NODE_COUNT, 2):
        raise ValueError(f"Invalid landmark cache: {cache_path}")

    aligned_cache[(video_id, label)] = aligned.astype(np.float64)

print("All 98 landmark caches loaded.")

def edge_distances(aligned):
    return np.linalg.norm(
        aligned[:, edges[:, 0], :]
        - aligned[:, edges[:, 1], :],
        axis=2
    )

def calculate_features(aligned, sigma):
    distances_by_frame = edge_distances(aligned)

    eigenvalue_frames = []

    for distances in distances_by_frame:
        weights = np.exp(
            -(distances ** 2) / (2.0 * sigma ** 2)
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
    jitter = (
        eigenvalues[2:]
        - 2.0 * eigenvalues[1:-1]
        + eigenvalues[:-2]
    )

    usable_eigenvalues = eigenvalues[:, 1:]
    usable_jitter = jitter[:, 1:]

    features = {}

    for i in range(usable_eigenvalues.shape[1]):
        col = usable_eigenvalues[:, i]
        name = i + 2
        features[f"eigen_{name}_mean"] = float(np.mean(col))
        features[f"eigen_{name}_std"] = float(np.std(col))
        features[f"eigen_{name}_min"] = float(np.min(col))
        features[f"eigen_{name}_max"] = float(np.max(col))

    for i in range(usable_jitter.shape[1]):
        col = usable_jitter[:, i]
        name = i + 2
        features[f"jitter_{name}_mean"] = float(np.mean(col))
        features[f"jitter_{name}_std"] = float(np.std(col))
        features[f"jitter_{name}_mean_abs"] = float(np.mean(np.abs(col)))
        features[f"jitter_{name}_max_abs"] = float(np.max(np.abs(col)))

    features["overall_jitter_mean_abs"] = float(
        np.mean(np.abs(usable_jitter))
    )
    features["overall_jitter_std"] = float(np.std(usable_jitter))

    return features

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

cv = GroupKFold(n_splits=FOLDS)

original_predictions = np.full(len(df), -1, dtype=int)
fixed_predictions = np.full(len(df), -1, dtype=int)

original_scores = np.full(len(df), np.nan)
fixed_scores = np.full(len(df), np.nan)

fold_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(df, y, groups), start=1
):
    print("\n" + "-" * 70)
    print(f"FOLD {fold}/{FOLDS}")

    train_ids = set(groups[train_idx])
    test_ids = set(groups[test_idx])

    if train_ids & test_ids:
        raise RuntimeError("Pair leakage detected.")

    print(
        f"Training videos: {len(train_idx)} | "
        f"Testing videos: {len(test_idx)}"
    )

    # Calculate common sigma using training videos only.
    training_distances = []

    for index in train_idx:
        row = df.iloc[index]
        aligned = aligned_cache[(row["video_id"], row["label"])]
        training_distances.append(edge_distances(aligned).ravel())

    fold_sigma = float(
        np.median(np.concatenate(training_distances))
    )

    print(f"Training-only sigma: {fold_sigma:.10f}")

    # Recalculate fixed-sigma features for every video in this fold.
    fixed_rows = []

    for position, index in enumerate(range(len(df)), start=1):
        row = df.iloc[index]
        aligned = aligned_cache[(row["video_id"], row["label"])]

        features = calculate_features(aligned, fold_sigma)
        fixed_rows.append(features)

    fixed_X = pd.DataFrame(fixed_rows)[feature_columns]
    original_X = df[feature_columns].astype(float)

    # Original per-video sigma model.
    original_model = make_model()
    original_model.fit(original_X.iloc[train_idx], y[train_idx])

    pred_original = original_model.predict(original_X.iloc[test_idx])
    score_original = original_model.decision_function(
        original_X.iloc[test_idx]
    )

    # Fixed training-only sigma model.
    fixed_model = make_model()
    fixed_model.fit(fixed_X.iloc[train_idx], y[train_idx])

    pred_fixed = fixed_model.predict(fixed_X.iloc[test_idx])
    score_fixed = fixed_model.decision_function(
        fixed_X.iloc[test_idx]
    )

    original_predictions[test_idx] = pred_original
    fixed_predictions[test_idx] = pred_fixed

    original_scores[test_idx] = score_original
    fixed_scores[test_idx] = score_fixed

    fold_rows.append({
        "fold": fold,
        "training_pairs": len(train_ids),
        "testing_pairs": len(test_ids),
        "training_sigma": fold_sigma,
        "original_accuracy": accuracy_score(
            y[test_idx], pred_original
        ),
        "fixed_accuracy": accuracy_score(
            y[test_idx], pred_fixed
        )
    })

    print(
        f"Per-video sigma accuracy: "
        f"{accuracy_score(y[test_idx], pred_original):.4f}"
    )
    print(
        f"Fixed sigma accuracy: "
        f"{accuracy_score(y[test_idx], pred_fixed):.4f}"
    )

def metrics(name, predictions, scores):
    return {
        "method": name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores)
    }

summary = pd.DataFrame([
    metrics("Per-video sigma", original_predictions, original_scores),
    metrics("Fold-specific training sigma", fixed_predictions, fixed_scores)
])

fold_df = pd.DataFrame(fold_rows)

print("\n" + "=" * 70)
print("FIVE-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print("\nFold-wise accuracy:")
print(fold_df.to_string(index=False))

print("\nPer-video sigma confusion matrix:")
print(confusion_matrix(y, original_predictions))

print("\nFixed-sigma confusion matrix:")
print(confusion_matrix(y, fixed_predictions))

summary.to_csv(
    os.path.join(OUT_DIR, "summary.csv"), index=False
)
fold_df.to_csv(
    os.path.join(OUT_DIR, "fold_results.csv"), index=False
)

pred_df = df[["video_id", "video_name", "label"]].copy()
pred_df["actual"] = np.where(y == 1, "fake", "real")
pred_df["per_video_prediction"] = np.where(
    original_predictions == 1, "fake", "real"
)
pred_df["fixed_sigma_prediction"] = np.where(
    fixed_predictions == 1, "fake", "real"
)
pred_df["per_video_score"] = original_scores
pred_df["fixed_sigma_score"] = fixed_scores

pred_df.to_csv(
    os.path.join(OUT_DIR, "predictions.csv"), index=False
)

print("\nSaved results in:", OUT_DIR)
print("=" * 70)
print("FIVE-FOLD EXPERIMENT COMPLETE")
print("=" * 70)
