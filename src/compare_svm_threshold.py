import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

df = pd.read_csv("outputs/features/uadfv_features.csv")
threshold = pd.read_csv(
    "outputs/eigen_jitter_threshold/nested_cv_predictions.csv"
)

# Normalize matching keys
for data in [df, threshold]:
    data["video_id"] = data["video_id"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()

# Select all 154 original spectral features
feature_cols = [
    c for c in df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c in ["overall_jitter_mean_abs", "overall_jitter_std"]
]

print("Spectral features:", len(feature_cols))

if len(feature_cols) != 154:
    raise ValueError(f"Expected 154 features, got {len(feature_cols)}")

# Merge using BOTH video_id and label
merged = df.merge(
    threshold[["video_id", "label", "prediction"]],
    on=["video_id", "label"],
    how="inner",
    validate="one_to_one"
)

print("Matched rows:", len(merged))

if len(merged) != 98:
    raise ValueError(f"Expected 98 matched videos, got {len(merged)}")

if merged.duplicated(["video_id", "label"]).any():
    raise ValueError("Duplicate video_id-label combinations found.")

X = merged[feature_cols].values
y = (merged["label"] == "fake").astype(int).values
groups = merged["video_id"].values

# Same five pair-aware folds for SVM
cv = GroupKFold(n_splits=5)

svm_predictions = np.zeros(len(merged), dtype=int)
svm_scores = np.zeros(len(merged), dtype=float)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])

    model.fit(X[train_idx], y[train_idx])
    svm_predictions[test_idx] = model.predict(X[test_idx])
    svm_scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    print(f"Fold {fold}: {len(test_idx)} test videos")

threshold_predictions = merged["prediction"].astype(int).values

def evaluate(name, predictions, scores=None):
    print(f"\n{name}")
    print("-" * 40)
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y, predictions), 4))
    print("Precision:", round(precision_score(y, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y, predictions, zero_division=0), 4))
    print("F1:", round(f1_score(y, predictions, zero_division=0), 4))

    if scores is not None:
        print("ROC-AUC:", round(roc_auc_score(y, scores), 4))

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

evaluate("SVM - 154 spectral features", svm_predictions, svm_scores)
evaluate("Nested Eigen-Jitter threshold", threshold_predictions)

print("\nPrediction agreement")
print("-" * 40)
print("Both correct:", np.sum((svm_predictions == y) & (threshold_predictions == y)))
print("Only SVM correct:", np.sum((svm_predictions == y) & (threshold_predictions != y)))
print("Only threshold correct:", np.sum((svm_predictions != y) & (threshold_predictions == y)))
print("Both wrong:", np.sum((svm_predictions != y) & (threshold_predictions != y)))

merged["actual"] = y
merged["svm_prediction"] = svm_predictions
merged["svm_fake_probability"] = svm_scores
merged["threshold_prediction"] = threshold_predictions
merged["svm_correct"] = svm_predictions == y
merged["threshold_correct"] = threshold_predictions == y

output = "outputs/eigen_jitter_threshold/svm_vs_jitter_predictions.csv"
merged.to_csv(output, index=False)

print("\nSaved comparison:", output)
