import os
import sys
import traceback
import warnings

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from scipy.linalg import eigh

warnings.filterwarnings("ignore")

ROOT = os.getcwd()
MANIFEST = os.path.join(ROOT, "outputs", "dataset", "uadfv_unique_manifest.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_OUTPUT = os.path.join(OUTPUT_DIR, "uadfv_features.csv")
FAILURE_OUTPUT = os.path.join(OUTPUT_DIR, "uadfv_failures.csv")

SAMPLE_FPS = 5
MAX_FRAMES = 150
EIGEN_COUNT = 20
MIN_VALID_FRAMES = 5

# Load clean dataset manifest
manifest = pd.read_csv(MANIFEST)

# Remove invalid entries defensively
manifest = manifest[
    manifest["Label"].isin(["real", "fake"])
].copy()

print("=" * 65)
print("UADFV BATCH SPECTRAL FEATURE EXTRACTION")
print("=" * 65)
print(f"Videos to process: {len(manifest)}")
print(f"Sampling rate: {SAMPLE_FPS} FPS")
print(f"Maximum frames per video: {MAX_FRAMES}")
print(f"Eigenvalues per frame: {EIGEN_COUNT}")

# Facial graph topology
connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
edges = np.array(sorted([tuple(sorted(edge)) for edge in connections]))
edges = np.unique(edges, axis=0)

NODE_COUNT = 468

if edges.size == 0:
    raise RuntimeError("Could not load MediaPipe face graph topology.")

print(f"Graph nodes: {NODE_COUNT}")
print(f"Graph edges: {len(edges)}")

def align_landmarks(points, reference):
    """Remove translation, scale and rotation using Procrustes alignment."""
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
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    frame_interval = max(1, round(fps / SAMPLE_FPS))

    landmark_frames = []
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

    # Edge lengths from all aligned frames.
    edge_distances = np.linalg.norm(
        aligned[:, edges[:, 0], :] - aligned[:, edges[:, 1], :],
        axis=2
    )

    sigma = float(np.median(edge_distances))

    if not np.isfinite(sigma) or sigma <= 1e-10:
        raise ValueError("Invalid graph bandwidth.")

    eigenvalue_frames = []

    for frame_number, points in enumerate(aligned):

        distances = np.linalg.norm(
            points[edges[:, 0]] - points[edges[:, 1]],
            axis=1
        )

        weights = np.exp(
            -(distances ** 2) / (2.0 * sigma ** 2)
        )

        adjacency = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)

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

    # Second temporal difference: lambda(t+1)-2*lambda(t)+lambda(t-1)
    jitter = (
        eigenvalues[2:]
        - 2.0 * eigenvalues[1:-1]
        + eigenvalues[:-2]
    )

    # Exclude first eigenvalue: approximately zero for a connected graph.
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

for index, row in manifest.iterrows():

    label = str(row["Label"])
    video_id = str(row["VideoID"])
    video_name = str(row["VideoName"])
    video_path = str(row["VideoPath"])

    print(
        f"\n[{len(successful_rows) + len(failure_rows) + 1}/{len(manifest)}] "
        f"{label.upper()} | {video_name}",
        flush=True
    )

    try:
        landmark_frames = extract_landmark_frames(video_path)

        features = calculate_spectral_features(landmark_frames)

        output_row = {
            "video_id": video_id,
            "video_name": video_name,
            "label": label,
            **features
        }

        successful_rows.append(output_row)

        print(
            f"SUCCESS | Valid frames: {features['frame_count']}",
            flush=True
        )

    except Exception as error:
        failure_rows.append({
            "video_id": video_id,
            "video_name": video_name,
            "label": label,
            "error": str(error)
        })

        print(f"FAILED | {error}", flush=True)

    # Save progress after every video
    if successful_rows:
        pd.DataFrame(successful_rows).to_csv(
            FEATURE_OUTPUT, index=False
        )

    pd.DataFrame(failure_rows).to_csv(
        FAILURE_OUTPUT, index=False
    )

# Final results
print("\n" + "=" * 65)
print("BATCH FEATURE EXTRACTION COMPLETE")
print("=" * 65)

if successful_rows:
    final_df = pd.DataFrame(successful_rows)
    final_df.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Successful videos: {len(final_df)}")
    print(f"Real videos: {(final_df['label'] == 'real').sum()}")
    print(f"Fake videos: {(final_df['label'] == 'fake').sum()}")
    print(f"Feature columns: {len(final_df.columns) - 3}")
    print(f"Feature CSV: {FEATURE_OUTPUT}")
else:
    print("No videos were successfully processed.")

print(f"Failed videos: {len(failure_rows)}")
print(f"Failure report: {FAILURE_OUTPUT}")
print("\nProcessing finished.")
