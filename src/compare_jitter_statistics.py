import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

df = pd.read_csv(
    "outputs/eigen_jitter_threshold/video_jitter_scores.csv"
)

y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].astype(str).to_numpy()

statistics = [
    "jitter_mean",
    "jitter_median",
    "jitter_std",
    "jitter_p95",
    "jitter_max"
]

cv = GroupKFold(n_splits=5)
results = []

def choose_threshold(values, labels):
    unique = np.unique(values)

    if len(unique) == 1:
        thresholds = unique
    else:
        thresholds = np.r_[
            unique[0] - 1e-12,
            (unique[:-1] + unique[1:]) / 2,
            unique[-1] + 1e-12
        ]

    best = None

    for direction in ["higher_is_fake", "lower_is_fake"]:
        for threshold in thresholds:
            if direction == "higher_is_fake":
                pred = (values >= threshold).astype(int)
            else:
                pred = (values <= threshold).astype(int)

            score = balanced_accuracy_score(labels, pred)

            if best is None or score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "score": float(score)
                }

    return best

for feature in statistics:
    values = df[feature].to_numpy(dtype=float)
    predictions = np.full(len(df), -1, dtype=int)

    for train_idx, test_idx in cv.split(df, y, groups):
        chosen = choose_threshold(
            values[train_idx],
            y[train_idx]
        )

        if chosen["direction"] == "higher_is_fake":
            predictions[test_idx] = (
                values[test_idx] >= chosen["threshold"]
            ).astype(int)
        else:
            predictions[test_idx] = (
                values[test_idx] <= chosen["threshold"]
            ).astype(int)

    auc = roc_auc_score(y, values)

    results.append({
        "statistic": feature,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "raw_auc": auc,
        "direction_adjusted_auc": max(auc, 1 - auc)
    })

result_df = pd.DataFrame(results).sort_values(
    "balanced_accuracy",
    ascending=False
)

result_df.to_csv(
    "outputs/eigen_jitter_threshold/statistic_comparison.csv",
    index=False
)

print("\nJITTER STATISTIC COMPARISON")
print(result_df.to_string(index=False))
print("\nSaved statistic_comparison.csv")
