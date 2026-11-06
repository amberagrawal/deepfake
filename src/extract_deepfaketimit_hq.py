import os
import warnings
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.linalg import eigh

warnings.filterwarnings("ignore")

ROOT = Path.cwd()

MANIFEST = ROOT / "outputs/dataset/deepfaketimit_hq_matching.csv"
MODEL_PATH = ROOT / "models/spectral_svm.joblib"

OUTPUT_DIR = ROOT / "outputs/features/deepfaketimit_hq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUTPUT = OUTPUT_DIR / "deepfaketimit_hq_features.csv"
FAILURE_OUTPUT = OUTPUT_DIR / "deepfaketimit_hq_failures.csv"

SAMPLE_FPS = 5
MAX_FRAMES = 150
EIGEN_COUNT = 20
MIN_VALID_FRAMES = 5
NODE_COUNT = 468

print("=" * 70)
print("DEEPFAKETIMIT HQ - EXTERNAL FEATURE EXTRACTION")
print("=" * 70)

# Load the original UADFV-trained model only to verify feature compatibility.
model_data = joblib.load(MODEL_PATH)
expected_features = model_data["feature_columns"]

if len(expected_features) != 154:
    raise RuntimeError(
        f"Expected 154 model features, found {len(expected_features)}"
    )

print("Loaded saved UADFV model.")
print("Expected spectral features:", len(expected_features))
print("Sampling FPS:", SAMPLE_FPS)
print("Maximum sampled frames:", MAX_FRAMES)

manifest = pd.read_csv(MANIFEST)

if len(manifest) != 320:
    raise RuntimeError(
        f"Expected 320 matched pairs, found {len(manifest)}"
    )

# Convert every pair into two independent video-processing jobs.
jobs = []

for _, row in manifest.iterrows():
    pair_id = f"{row['TargetIdentity']}_{row['Utterance']}"

    jobs.append({
        "pair_id": pair_id,
        "target_identity": row["TargetIdentity"],
        "swapped_identity": row["SwappedIdentity"],
        "utterance": row["Utterance"],
        "label": "real",
        "video_path": row["RealVideoPath"],
    })

    jobs.append({
        "pair_id": pair_id,
        "target_identity": row["TargetIdentity"],
        "swapped_identity": row["SwappedIdentity"],
        "utterance": row["Utterance"],
        "label": "fake",
        "video_path": row["FakeVideoPath"],
    })

if len(jobs) != 640:
    raise RuntimeError(f"Expected 640 videos, found {len(jobs)}")

# Original MediaPipe facial graph topology.
connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
edges = np.array(
    sorted([tuple(sorted(edge)) for edge in connections])
)
edges = np.unique(edges, axis=0)

if len(edges) != 1322:
    raise RuntimeError(
        f"Expected 1322 graph edges, found {len(edges)}"
    )

print("Graph nodes:", NODE_COUNT)
print("Graph edges:", len(edges))
print("Videos to process:", len(jobs))


def align_landmarks(points, reference):
    """Same Procrustes alignment used in the UADFV extractor."""

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


def extract_landmark_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    frame_interval = max(1, round(fps / SAMPLE_FPS))

    landmark_frames = []
    frame_index = 0
    sampled_count = 0

    try:
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

                if current_index % frame_interval != 0:
                    continue

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                sampled_count += 1

                if not result.multi_face_landmarks:
                    continue

                face = result.multi_face_landmarks[0]

                points = np.array([
                    [landmark.x * width, landmark.y * height]
                    for landmark in face.landmark
                ], dtype=np.float64)

                if points.shape != (NODE_COUNT, 2):
                    continue

                if not np.isfinite(points).all():
                    continue

                landmark_frames.append(points)

    finally:
        cap.release()

    return landmark_frames


def calculate_spectral_features(landmark_frames):
    if len(landmark_frames) < MIN_VALID_FRAMES:
        raise ValueError(
            f"Only {len(landmark_frames)} valid face frames detected."
        )

    reference = landmark_frames[0]
    aligned = []

    for points in landmark_frames:
        aligned.append(align_landmarks(points, reference))

    aligned = np.asarray(aligned)

    # Same per-video median graph bandwidth as UADFV.
    edge_distances = np.linalg.norm(
        aligned[:, edges[:, 0], :] -
        aligned[:, edges[:, 1], :],
        axis=2
    )

    sigma = float(np.median(edge_distances))

    if not np.isfinite(sigma) or sigma <= 1e-10:
        raise ValueError("Invalid graph bandwidth.")

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
            raise ValueError("Graph contains an isolated node.")

        inverse_sqrt_degree = 1.0 / np.sqrt(degree)

        normalized_adjacency = (
            inverse_sqrt_degree[:, None]
            * adjacency
            * inverse_sqrt_degree[None, :]
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

    # Exclude first eigenvalue, exactly as in the original extractor.
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
        features[f"jitter_{name}_mean_abs"] = float(
            np.mean(np.abs(column))
        )
        features[f"jitter_{name}_max_abs"] = float(
            np.max(np.abs(column))
        )

    features["overall_jitter_mean_abs"] = float(
        np.mean(np.abs(usable_jitter))
    )

    features["overall_jitter_std"] = float(
        np.std(usable_jitter)
    )

    features["frame_count"] = len(landmark_frames)
    features["sigma"] = sigma

    if not np.isfinite(list(features.values())).all():
        raise ValueError("Non-finite feature values generated.")

    return features


successful_rows = []
failure_rows = []

for index, job in enumerate(jobs, start=1):

    video_path = Path(job["video_path"])

    print(
        f"\n[{index}/{len(jobs)}] "
        f"{job['label'].upper()} | "
        f"{job['target_identity']} | "
        f"{job['utterance']}",
        flush=True
    )

    try:
        if not video_path.exists():
            raise FileNotFoundError(str(video_path))

        landmark_frames = extract_landmark_frames(video_path)
        features = calculate_spectral_features(landmark_frames)

        output_row = {
            "pair_id": job["pair_id"],
            "target_identity": job["target_identity"],
            "swapped_identity": job["swapped_identity"],
            "utterance": job["utterance"],
            "label": job["label"],
            "video_name": video_path.name,
            "video_path": str(video_path),
            **features
        }

        successful_rows.append(output_row)

        print(
            f"SUCCESS | Valid frames: {features['frame_count']} "
            f"| Features: 154",
            flush=True
        )

    except Exception as error:
        failure_rows.append({
            "pair_id": job["pair_id"],
            "target_identity": job["target_identity"],
            "label": job["label"],
            "video_path": str(video_path),
            "error": str(error)
        })

        print(f"FAILED | {error}", flush=True)

    # Save progress after every video.
    pd.DataFrame(successful_rows).to_csv(
        FEATURE_OUTPUT, index=False
    )

    pd.DataFrame(failure_rows).to_csv(
        FAILURE_OUTPUT, index=False
    )

print("\n" + "=" * 70)
print("EXTERNAL FEATURE EXTRACTION COMPLETE")
print("=" * 70)

if successful_rows:
    final_df = pd.DataFrame(successful_rows)
    print("Successful videos:", len(final_df))
    print("Real videos:", (final_df["label"] == "real").sum())
    print("Fake videos:", (final_df["label"] == "fake").sum())
    print("Total columns:", len(final_df.columns))
    print("Spectral feature count:", len(expected_features))
    print("Feature CSV:", FEATURE_OUTPUT)

print("Failed videos:", len(failure_rows))
print("Failure CSV:", FAILURE_OUTPUT)

print("\nModel was not trained or modified.")
print("Processing finished.")
