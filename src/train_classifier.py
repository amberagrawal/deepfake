import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "outputs/features/uadfv_features.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("PAIR-AWARE SVM CLASSIFICATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

# Use only spectral measurements, not video metadata.
feature_columns = [
    c for c in df.columns
    if c.startswith("eigen_")
    or c.startswith("jitter_")
    or c.startswith("overall_jitter_")
]

X = df[feature_columns].astype(float)
y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].astype(str).to_numpy()

if len(feature_columns) == 0:
    raise ValueError("No spectral feature columns found.")

if np.isnan(X.to_numpy()).any() or not np.isfinite(X.to_numpy()).all():
    raise ValueError("Features contain missing or non-finite values.")

# Verify every group contains exactly one real and one fake video.
pair_check = df.groupby("video_id")["label"].agg(
    lambda values: set(values)
)
if not all(value == {"real", "fake"} for value in pair_check):
    raise ValueError("Some video ID groups do not contain a real/fake pair.")

print(f"Videos: {len(df)}")
print(f"Paired video IDs: {len(np.unique(groups))}")
print(f"Spectral features: {len(feature_columns)}")
print("Real = 0 | Fake = 1")

cv = GroupKFold(n_splits=5)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

print("\nRunning 5-fold pair-aware cross-validation...")

# Out-of-fold predictions: every video is predicted by a model
# that was not trained on either member of its pair.
predictions = cross_val_predict(
    svm_pipeline, X, y,
    cv=cv,
    groups=groups,
    method="predict"
)

scores = cross_val_predict(
    svm_pipeline, X, y,
    cv=cv,
    groups=groups,
    method="decision_function"
)

# Majority-class baseline.
baseline = DummyClassifier(strategy="most_frequent")
baseline_predictions = cross_val_predict(
    baseline, X, y,
    cv=cv,
    groups=groups,
    method="predict"
)

print("\n" + "=" * 65)
print("CROSS-VALIDATION RESULTS")
print("=" * 65)

print(f"SVM accuracy:  {accuracy_score(y, predictions):.4f}")
print(f"Baseline accuracy: {accuracy_score(y, baseline_predictions):.4f}")
print(f"Precision:     {precision_score(y, predictions, zero_division=0):.4f}")
print(f"Recall:        {recall_score(y, predictions, zero_division=0):.4f}")
print(f"F1-score:      {f1_score(y, predictions, zero_division=0):.4f}")
print(f"ROC-AUC:       {roc_auc_score(y, scores):.4f}")

print("\nClassification report:")
print(classification_report(
    y, predictions,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y, predictions, labels=[0, 1]))

# Show accuracy for each held-out fold.
print("\nFold-wise accuracy:")
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    fold_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])
    fold_model.fit(X.iloc[train_idx], y[train_idx])
    fold_predictions = fold_model.predict(X.iloc[test_idx])
    print(
        f"Fold {fold}: "
        f"{accuracy_score(y[test_idx], fold_predictions):.4f} "
        f"({len(test_idx)} videos)"
    )

# Save out-of-fold predictions for later analysis.
prediction_df = df[["video_id", "video_name", "label"]].copy()
prediction_df["actual"] = np.where(y == 1, "fake", "real")
prediction_df["predicted"] = np.where(
    predictions == 1, "fake", "real"
)
prediction_df["decision_score_fake"] = scores
prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "svm_cv_predictions.csv"),
    index=False
)

# Train the final model on all available videos for later experiments.
svm_pipeline.fit(X, y)

joblib.dump(
    {
        "model": svm_pipeline,
        "feature_columns": feature_columns,
        "class_mapping": {"real": 0, "fake": 1}
    },
    os.path.join(MODEL_DIR, "spectral_svm.joblib")
)

print("\nSaved model: models/spectral_svm.joblib")
print("Saved predictions: outputs/evaluation/svm_cv_predictions.csv")
print("\n" + "=" * 65)
print("CLASSIFICATION EXPERIMENT COMPLETE")
print("=" * 65)
