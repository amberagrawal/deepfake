import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from scipy.linalg import eigvalsh


INPUT_CSV = "outputs/landmarks/landmarks.csv"
OUTPUT_DIR = "outputs/spectral"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "spectral_features.csv")

NUM_EIGENVALUES = 20


def load_landmarks(csv_path):
    df = pd.read_csv(csv_path)

    required = {"frame", "landmark_index", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )

    frames = []

    for frame_id, group in df.groupby("frame"):
        group = group.sort_values("landmark_index")

        points = group[["x", "y", "z"]].to_numpy(dtype=np.float64)

        if len(points) != 468:
            continue

        frames.append((int(frame_id), points))

    if len(frames) < 3:
        raise ValueError("At least 3 valid face frames are required.")

    return frames


def procrustes_align(points, reference):
    """
    Align one 3D facial landmark configuration to a reference.
    Removes translation, scale, and rotation.
    """

    source = points.copy()
    target = reference.copy()

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source -= source_center
    target -= target_center

    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)

    if source_norm < 1e-12 or target_norm < 1e-12:
        return source

    source /= source_norm
    target /= target_norm

    U, _, Vt = np.linalg.svd(source.T @ target)

    rotation = U @ Vt

    # Avoid reflection.
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt

    aligned = source @ rotation

    # Match reference scale.
    aligned *= target_norm

    return aligned


def create_face_edges():
    connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

    edges = set()

    for start, end in connections:
        if start != end:
            edges.add((min(start, end), max(start, end)))

    return sorted(edges)


def calculate_graph_spectrum(points, edges, sigma):
    """
    Create a dynamic weighted facial graph and calculate
    the smallest eigenvalues of its normalized Laplacian.
    """

    number_of_nodes = len(points)
    adjacency = np.zeros(
        (number_of_nodes, number_of_nodes),
        dtype=np.float64
    )

    for start, end in edges:
        distance = np.linalg.norm(points[start] - points[end])

        weight = np.exp(
            -(distance ** 2) / (2.0 * sigma ** 2)
        )

        adjacency[start, end] = weight
        adjacency[end, start] = weight

    degrees = adjacency.sum(axis=1)

    inverse_sqrt_degree = np.zeros_like(degrees)
    valid = degrees > 1e-12
    inverse_sqrt_degree[valid] = 1.0 / np.sqrt(degrees[valid])

    normalized_adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )

    laplacian = np.eye(number_of_nodes) - normalized_adjacency

    eigenvalues = eigvalsh(
        laplacian,
        subset_by_index=[0, NUM_EIGENVALUES - 1],
        driver="evr"
    )

    return eigenvalues


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading facial landmarks...")

    frames = load_landmarks(INPUT_CSV)

    print(f"Valid frames: {len(frames)}")

    reference = frames[0][1]

    print("\nApplying Procrustes alignment...")

    aligned_frames = []

    for frame_id, points in frames:
        aligned = procrustes_align(points, reference)
        aligned_frames.append((frame_id, aligned))

    print("Alignment completed.")

    print("\nCreating facial graph...")

    edges = create_face_edges()

    print(f"Number of facial graph nodes: 468")
    print(f"Number of facial graph edges: {len(edges)}")

    # Use one fixed bandwidth for the entire video.
    all_distances = []

    for _, points in aligned_frames:
        for start, end in edges:
            distance = np.linalg.norm(points[start] - points[end])
            all_distances.append(distance)

    sigma = float(np.median(all_distances))

    if sigma < 1e-12:
        raise ValueError("Invalid graph bandwidth.")

    print(f"Graph bandwidth: {sigma:.8f}")

    print("\nCalculating Graph Laplacian eigenvalues...")

    records = []

    for index, (frame_id, points) in enumerate(aligned_frames):

        eigenvalues = calculate_graph_spectrum(
            points,
            edges,
            sigma
        )

        record = {"frame": frame_id}

        for eigen_index, value in enumerate(eigenvalues):
            record[f"eigen_{eigen_index + 1}"] = float(value)

        records.append(record)

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(aligned_frames)} frames")

    df = pd.DataFrame(records)

    print("\nCalculating temporal eigen-jitter...")

    eigen_columns = [
        f"eigen_{i + 1}" for i in range(NUM_EIGENVALUES)
    ]

    for column in eigen_columns:
        values = df[column].to_numpy()

        jitter = np.full(len(values), np.nan)

        jitter[1:-1] = (
            values[2:] - 2 * values[1:-1] + values[:-2]
        )

        df[f"jitter_{column}"] = jitter

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n========== PHASE 3 COMPLETE ==========")
    print(f"Frames processed: {len(df)}")
    print(f"Graph nodes: 468")
    print(f"Graph edges: {len(edges)}")
    print(f"Eigenvalues per frame: {NUM_EIGENVALUES}")
    print(f"Output: {OUTPUT_CSV}")

    print("\nPreview of spectral features:")
    print(df.head())


if __name__ == "__main__":
    main()
