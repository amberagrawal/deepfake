import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

INPUT = "outputs/eigen_jitter_threshold/video_jitter_scores.csv"
OUT = "outputs/eigen_jitter_threshold/nested_cv_results.csv"

df = pd.read_csv(INPUT)

y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].astype(str).to_numpy()

features = [
    "jitter_mean",
    "jitter_median",
    "jitter_std",
    "jitter_p95",
    "jitter_max"
]

def fit_threshold(values, labels):
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

def predict(values, threshold, direction):
    if direction == "higher_is_fake":
        return (values >= threshold).astype(int)
    return (values <= threshold).astype(int)

outer_cv = GroupKFold(n_splits=5)

oof_predictions = np.full(len(df), -1, dtype=int)
oof_scores = np.full(len(df), np.nan)
fold_results = []

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(df, y, groups), start=1
):
    X_train = df.iloc[train_idx]
    y_train = y[train_idx]
    g_train = groups[train_idx]

    inner_cv = GroupKFold(n_splits=4)

    feature_scores = {}

    # Select statistic using inner CV only.
    for feature in features:
        inner_predictions = np.full(len(train_idx), -1, dtype=int)

        for inner_train, inner_val in inner_cv.split(
            X_train, y_train, g_train
        ):
            train_values = X_train.iloc[inner_train][feature].to_numpy()
            val_values = X_train.iloc[inner_val][feature].to_numpy()

            chosen = fit_threshold(
                train_values, y_train[inner_train]
            )

            inner_predictions[inner_val] = predict(
                val_values,
                chosen["threshold"],
                chosen["direction"]
            )

        feature_scores[feature] = balanced_accuracy_score(
            y_train, inner_predictions
        )

    selected_feature = max(
        feature_scores,
        key=feature_scores.get
    )

    # Calibrate final threshold using the complete outer training set.
    chosen = fit_threshold(
        X_train[selected_feature].to_numpy(),
        y_train
    )

    test_values = df.iloc[test_idx][selected_feature].to_numpy()

    predictions = predict(
        test_values,
        chosen["threshold"],
        chosen["direction"]
    )

    oof_predictions[test_idx] = predictions

    # Orient raw scores so larger means more likely fake.
    if chosen["direction"] == "higher_is_fake":
        oof_scores[test_idx] = test_values
    else:
        oof_scores[test_idx] = -test_values

    fold_results.append({
        "fold": outer_fold,
        "selected_feature": selected_feature,
        "inner_balanced_accuracy": feature_scores[selected_feature],
        "threshold": chosen["threshold"],
        "direction": chosen["direction"],
        "test_accuracy": accuracy_score(y[test_idx], predictions),
        "test_balanced_accuracy": balanced_accuracy_score(
            y[test_idx], predictions
        ),
        "test_pairs": len(np.unique(groups[test_idx]))
    })

    print(
        f"Outer fold {outer_fold}: selected={selected_feature}, "
        f"test accuracy={accuracy_score(y[test_idx], predictions):.4f}"
    )

print("\nNESTED CROSS-VALIDATION RESULTS")
print("Accuracy:", accuracy_score(y, oof_predictions))
print("Balanced accuracy:", balanced_accuracy_score(y, oof_predictions))
print("Precision:", precision_score(y, oof_predictions, zero_division=0))
print("Recall:", recall_score(y, oof_predictions, zero_division=0))
print("F1:", f1_score(y, oof_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y, oof_scores))

print("\nFold details:")
fold_df = pd.DataFrame(fold_results)
print(fold_df.to_string(index=False))

fold_df.to_csv(OUT, index=False)

pred_df = df[["video_id", "label"]].copy()
pred_df["prediction"] = oof_predictions
pred_df["oriented_score"] = oof_scores
pred_df.to_csv(
    "outputs/eigen_jitter_threshold/nested_cv_predictions.csv",
    index=False
)

print("\nSaved nested CV results.")
