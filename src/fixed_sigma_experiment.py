import os
import warnings
import joblib
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from scipy.linalg import eigh
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = os.getcwd()
MANIFEST_PATH = "outputs/dataset/uadfv_unique_manifest.csv"
ORIGINAL_FEATURES = "outputs/features/uadfv_features.csv"

OUT_DIR = "outputs/fixed_sigma"
CACHE_DIR = os.path.join(OUT_DIR, "aligned_landmarks")
MODEL_DIR = "models"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_FPS = 5
MAX_FRAMES = 150
NODE_COUNT = 468
EIGEN_COUNT = 20
MIN_VALID_FRAMES = 5

print("=" * 70)
print("FIXED-SIGMA EXPERIMENT")
print("=" * 70)

manifest = pd.read_csv(MANIFEST_PATH)
manifest["VideoID"] = manifest["VideoID"].astype(str)
manifest["Label"] = manifest["Label"].str.lower()

original = pd.read_csv(ORIGINAL_FEATURES)
original["video_id"] = original["video_id"].astype(str)

# Use the same deterministic pair-aware split for both experiments.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

y_all = original["label"].map({"real": 0, "fake": 1}).to_numpy()
groups_all = original["video_id"].to_numpy()

train_idx, test_idx = next(
    splitter.split(original, y_all, groups_all)
)

train_ids = set(groups_all[train_idx])
test_ids = set(groups_all[test_idx])

if train_ids & test_ids:
    raise RuntimeError("Pair leakage detected in the split.")

print(f"Training pairs: {len(train_ids)}")
print(f"Testing pairs: {len(test_ids)}")

split_df = pd.DataFrame({
    "video_id": sorted(train_ids) + sorted(test_ids),
    "split": (
        ["train"] * len(train_ids)
        + ["test"] * len(test_ids)
    )
})
split_df.to_csv(os.path.join(OUT_DIR, "pair_split.csv"), index=False)

edges = mp.solutions.face_mesh.FACEMESH_TESSELATION
edges = np.array(
    sorted([tuple(sorted(e)) for e in edges]),
    dtype=np.int32
)
edges = np.unique(edges, axis=0)

print(f"Graph nodes: {NODE_COUNT}")
print(f"Graph edges: {len(edges)}")

def align_landmarks(points, reference):
    points = points - points.mean(axis=0, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)

    norm_points = np.linalg.norm(points)
    norm_reference = np.linalg.norm(reference)

    if norm_points < 1e-10 or norm_reference < 1e-10:
        raise ValueError("Invalid landmark geometry.")

    points = points / norm_points
    reference = reference / norm_reference

    u, _, vt = np.linalg.svd(points.T @ reference)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    return points @ rotation

def extract_and_align(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    interval = max(1, round(fps / SAMPLE_FPS))
    raw_frames = []
    frame_index = 0
    sampled_count = 0

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while sampled_count < MAX_FRAMES:
            success, frame = cap.read()

            if not success:
                break

            current_index = frame_index
            frame_index += 1

            if current_index % interval != 0:
                continue

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            sampled_count += 1

            if not result.multi_face_landmarks:
                continue

            face = result.multi_face_landmarks[0]

            points = np.array([
                [p.x * width, p.y * height]
                for p in face.landmark
            ], dtype=np.float64)

            if points.shape == (NODE_COUNT, 2) and np.isfinite(points).all():
                raw_frames.append(points)

    cap.release()

    if len(raw_frames) < MIN_VALID_FRAMES:
        raise ValueError(
            f"Only {len(raw_frames)} valid face frames found."
        )

    reference = raw_frames[0]

    aligned = np.asarray([
        align_landmarks(points, reference)
        for points in raw_frames
    ], dtype=np.float32)

    return aligned

def get_cache_path(row):
    filename = f"{row['Label']}_{row['VideoID']}.npy"
    return os.path.join(CACHE_DIR, filename)

# Stage 1: Extract and cache aligned landmarks for all videos.
print("\nSTAGE 1: EXTRACTING ALIGNED LANDMARKS")
print("-" * 70)

for position, (_, row) in enumerate(manifest.iterrows(), start=1):
    cache_path = get_cache_path(row)

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.ndim == 3 and cached.shape[1:] == (NODE_COUNT, 2):
            print(f"[{position}/{len(manifest)}] CACHE | {row['VideoName']}")
            continue

    try:
        aligned = extract_and_align(str(row["VideoPath"]))
        np.save(cache_path, aligned)
        print(
            f"[{position}/{len(manifest)}] SUCCESS | "
            f"{row['VideoName']} | {len(aligned)} frames",
            flush=True
        )
    except Exception as error:
        raise RuntimeError(
            f"Landmark extraction failed for {row['VideoName']}: {error}"
        )

# Stage 2: Calculate one common sigma using training videos only.
print("\nSTAGE 2: CALCULATING TRAINING-ONLY SIGMA")
print("-" * 70)

training_distances = []

for _, row in manifest.iterrows():
    if str(row["VideoID"]) not in train_ids:
        continue

    aligned = np.load(get_cache_path(row)).astype(np.float64)

    distances = np.linalg.norm(
        aligned[:, edges[:, 0], :] - aligned[:, edges[:, 1], :],
        axis=2
    )

    training_distances.append(distances.ravel())

common_sigma = float(np.median(np.concatenate(training_distances)))

if not np.isfinite(common_sigma) or common_sigma <= 1e-10:
    raise RuntimeError("Invalid common sigma calculated.")

print(f"Common training sigma: {common_sigma:.10f}")

with open(os.path.join(OUT_DIR, "common_sigma.txt"), "w") as file:
    file.write(str(common_sigma))

# Stage 3: Generate spectral features with the common sigma.
def calculate_features(aligned, sigma):
    aligned = aligned.astype(np.float64)

    eigenvalue_frames = []

    for points in aligned:
        distances = np.linalg.norm(
            points[edges[:, 0]] - points[edges[:, 1]],
            axis=1
        )

        weights = np.exp(
            -(distances ** 2) / (2.0 * sigma ** 2)
        )

        adjacency = np.zeros(
            (NODE_COUNT, NODE_COUNT),
            dtype=np.float64
        )

        adjacency[edges[:, 0], edges[:, 1]] = weights
        adjacency[edges[:, 1], edges[:, 0]] = weights

        degree = adjacency.sum(axis=1)

        if np.any(degree <= 1e-12):
            raise ValueError("Graph contains isolated nodes.")

        inv_sqrt_degree = 1.0 / np.sqrt(degree)

        normalized_adjacency = (
            inv_sqrt_degree[:, None]
            * adjacency
            * inv_sqrt_degree[None, :]
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

    for index in range(usable_eigenvalues.shape[1]):
        column = usable_eigenvalues[:, index]
        name = index + 2

        features[f"eigen_{name}_mean"] = float(np.mean(column))
        features[f"eigen_{name}_std"] = float(np.std(column))
        features[f"eigen_{name}_min"] = float(np.min(column))
        features[f"eigen_{name}_max"] = float(np.max(column))

    for index in range(usable_jitter.shape[1]):
        column = usable_jitter[:, index]
        name = index + 2

        features[f"jitter_{name}_mean"] = float(np.mean(column))
        features[f"jitter_{name}_std"] = float(np.std(column))
        features[f"jitter_{name}_mean_abs"] = float(np.mean(np.abs(column)))
        features[f"jitter_{name}_max_abs"] = float(np.max(np.abs(column)))

    features["overall_jitter_mean_abs"] = float(
        np.mean(np.abs(usable_jitter))
    )
    features["overall_jitter_std"] = float(np.std(usable_jitter))
    features["frame_count"] = len(aligned)
    features["sigma"] = sigma

    if not np.isfinite(list(features.values())).all():
        raise ValueError("Non-finite feature values.")

    return features

print("\nSTAGE 3: GENERATING FIXED-SIGMA FEATURES")
print("-" * 70)

fixed_rows = []

for position, (_, row) in enumerate(manifest.iterrows(), start=1):
    aligned = np.load(get_cache_path(row))

    features = calculate_features(aligned, common_sigma)

    fixed_rows.append({
        "video_id": str(row["VideoID"]),
        "video_name": str(row["VideoName"]),
        "label": str(row["Label"]),
        **features
    })

    print(
        f"[{position}/{len(manifest)}] FEATURES READY | "
        f"{row['VideoName']}",
        flush=True
    )

fixed_df = pd.DataFrame(fixed_rows)

fixed_path = os.path.join(OUT_DIR, "fixed_sigma_features.csv")
fixed_df.to_csv(fixed_path, index=False)

print("\nFixed-sigma feature file saved:", fixed_path)

# Stage 4: Compare both methods on exactly the same test pairs.
feature_columns = [
    c for c in fixed_df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

original_feature_columns = [
    c for c in original.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

def evaluate_model(X, y, train_mask, test_mask):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X.loc[train_mask], y[train_mask])

    predictions = pipeline.predict(X.loc[test_mask])
    scores = pipeline.decision_function(X.loc[test_mask])

    y_test = y[test_mask]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }

    return pipeline, metrics, predictions, scores

# Align rows by video ID and label.
original["video_id"] = original["video_id"].astype(str)
fixed_df["video_id"] = fixed_df["video_id"].astype(str)

original = original.sort_values(
    ["video_id", "label"]
).reset_index(drop=True)

fixed_df = fixed_df.sort_values(
    ["video_id", "label"]
).reset_index(drop=True)

if not (
    original["video_id"].tolist() == fixed_df["video_id"].tolist()
    and original["label"].tolist() == fixed_df["label"].tolist()
):
    raise RuntimeError("Original and fixed feature rows do not match.")

train_mask = original["video_id"].isin(train_ids).to_numpy()
test_mask = original["video_id"].isin(test_ids).to_numpy()

y = original["label"].map({"real": 0, "fake": 1}).to_numpy()

X_original = original[original_feature_columns].astype(float)
X_fixed = fixed_df[feature_columns].astype(float)

print("\nSTAGE 4: TRAINING AND COMPARING MODELS")
print("-" * 70)

original_model, original_metrics, original_pred, original_scores = (
    evaluate_model(X_original, y, train_mask, test_mask)
)

fixed_model, fixed_metrics, fixed_pred, fixed_scores = (
    evaluate_model(X_fixed, y, train_mask, test_mask)
)

comparison = pd.DataFrame([
    {"method": "Per-video sigma", **{
        k: v for k, v in original_metrics.items()
        if k != "confusion_matrix"
    }},
    {"method": "Fixed training sigma", **{
        k: v for k, v in fixed_metrics.items()
        if k != "confusion_matrix"
    }}
])

print("\nTEST SET RESULTS")
print(comparison.to_string(index=False))

print("\nPer-video sigma confusion matrix:")
print(np.array(original_metrics["confusion_matrix"]))

print("\nFixed training sigma confusion matrix:")
print(np.array(fixed_metrics["confusion_matrix"]))

comparison.to_csv(
    os.path.join(OUT_DIR, "sigma_comparison.csv"),
    index=False
)

predictions_df = original[
    ["video_id", "video_name", "label"]
].copy()

predictions_df["split"] = np.where(train_mask, "train", "test")
predictions_df["original_prediction"] = ""
predictions_df["fixed_sigma_prediction"] = ""
predictions_df["original_score"] = np.nan
predictions_df["fixed_sigma_score"] = np.nan

predictions_df.loc[test_mask, "original_prediction"] = np.where(
    original_pred == 1, "fake", "real"
)
predictions_df.loc[test_mask, "fixed_sigma_prediction"] = np.where(
    fixed_pred == 1, "fake", "real"
)
predictions_df.loc[test_mask, "original_score"] = original_scores
predictions_df.loc[test_mask, "fixed_sigma_score"] = fixed_scores

predictions_df.to_csv(
    os.path.join(OUT_DIR, "test_predictions.csv"),
    index=False
)

joblib.dump(
    {
        "model": fixed_model,
        "feature_columns": feature_columns,
        "common_sigma": common_sigma,
        "class_mapping": {"real": 0, "fake": 1}
    },
    os.path.join(MODEL_DIR, "spectral_svm_fixed_sigma.joblib")
)

print("\nSaved files:")
print("- outputs/fixed_sigma/pair_split.csv")
print("- outputs/fixed_sigma/common_sigma.txt")
print("- outputs/fixed_sigma/fixed_sigma_features.csv")
print("- outputs/fixed_sigma/sigma_comparison.csv")
print("- outputs/fixed_sigma/test_predictions.csv")
print("- models/spectral_svm_fixed_sigma.joblib")

print("\n" + "=" * 70)
print("FIXED-SIGMA EXPERIMENT COMPLETE")
print("=" * 70)
