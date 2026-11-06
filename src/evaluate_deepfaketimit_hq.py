from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "spectral_svm.joblib"
FEATURE_PATH = ROOT / "outputs" / "features" / "deepfaketimit_hq" / "deepfaketimit_hq_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "deepfaketimit_hq"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DEEPFAKETIMIT HIGH-QUALITY EXTERNAL EVALUATION")
print("=" * 65)

# 1. Load the existing UADFV model
print("\n[1/6] Loading trained UADFV model...")
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_columns = model_data["feature_columns"]
class_mapping = model_data["class_mapping"]

print("Model loaded successfully.")
print("Expected spectral features:", len(feature_columns))
print("Class mapping:", class_mapping)

# 2. Load external dataset features
print("\n[2/6] Loading DeepfakeTIMIT features...")
df = pd.read_csv(FEATURE_PATH)

print("Videos found:", len(df))
print("Columns found:", len(df.columns))

if len(df) != 640:
    raise ValueError(f"Expected 640 videos, found {len(df)}")

if "label" not in df.columns:
    raise ValueError("The feature CSV does not contain a 'label' column.")

# 3. Validate labels and prepare features
print("\n[3/6] Validating labels and preparing features...")

df["label"] = df["label"].astype(str).str.strip().str.lower()

if set(df["label"].unique()) != {"real", "fake"}:
    raise ValueError(f"Unexpected labels: {df['label'].unique()}")

if df["label"].value_counts().to_dict() != {"real": 320, "fake": 320}:
    raise ValueError("Expected exactly 320 real and 320 fake videos.")

missing = [col for col in feature_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

# Use exactly the feature columns and order from the trained model
X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Feature matrix contains missing or non-finite values.")

y_true = df["label"].map(class_mapping).to_numpy()

print("Feature matrix shape:", X.shape)
print("Real videos:", int((y_true == 0).sum()))
print("Fake videos:", int((y_true == 1).sum()))

# 4. Predict without fitting or modifying the model
print("\n[4/6] Running predictions with the existing SVM...")

y_pred = model.predict(X)
scores = model.decision_function(X)

print("Predictions completed.")

# 5. Calculate evaluation metrics
print("\n[5/6] Calculating evaluation metrics...")

accuracy = accuracy_score(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
auc = roc_auc_score(y_true, scores)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\n" + "=" * 65)
print("EXTERNAL EVALUATION RESULTS")
print("=" * 65)

print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"ROC-AUC:           {auc:.4f}")

print("\nConfusion Matrix")
print("Rows = Actual, Columns = Predicted")
print("                 Predicted Real   Predicted Fake")
print(f"Actual Real      {cm[0, 0]:14d}   {cm[0, 1]:14d}")
print(f"Actual Fake      {cm[1, 0]:14d}   {cm[1, 1]:14d}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

# 6. Save predictions and summary
print("\n[6/6] Saving evaluation results...")

results = df.copy()
results["actual_label"] = df["label"]
results["predicted_label"] = np.where(y_pred == 0, "real", "fake")
results["correct_prediction"] = y_true == y_pred
results["svm_decision_score"] = scores

predictions_path = OUTPUT_DIR / "external_predictions.csv"
results.to_csv(predictions_path, index=False)

summary = {
    "dataset": "DeepfakeTIMIT_High_Quality",
    "total_videos": int(len(df)),
    "real_videos": int((y_true == 0).sum()),
    "fake_videos": int((y_true == 1).sum()),
    "features_used": int(len(feature_columns)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "precision_fake": float(precision),
    "recall_fake": float(recall),
    "f1_fake": float(f1),
    "roc_auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "model_retrained": False
}

summary_path = OUTPUT_DIR / "external_evaluation_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("Predictions saved to:", predictions_path)
print("Summary saved to:", summary_path)

print("\nModel was not retrained or modified.")
print("EXTERNAL EVALUATION COMPLETE")
