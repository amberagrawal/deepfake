import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

DATA_PATH = "outputs/features/uadfv_features.csv"
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

y = df["label"].map({"real": 0, "fake": 1}).to_numpy()
groups = df["video_id"].astype(str).to_numpy()

eigen_features = [
    c for c in df.columns if c.startswith("eigen_")
]

jitter_features = [
    c for c in df.columns
    if c.startswith("jitter_") or c.startswith("overall_jitter_")
]

feature_sets = {
    "Eigenvalue only": eigen_features,
    "Eigen-jitter only": jitter_features,
    "Combined": eigen_features + jitter_features
}

cv = GroupKFold(n_splits=5)
results = []

print("=" * 65)
print("FEATURE ABLATION EXPERIMENT")
print("=" * 65)
print("Eigenvalue features:", len(eigen_features))
print("Jitter features:", len(jitter_features))
print("Combined features:", len(eigen_features) + len(jitter_features))

for model_name, columns in feature_sets.items():
    X = df[columns].astype(float)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    predictions = cross_val_predict(
        pipeline, X, y,
        cv=cv, groups=groups,
        method="predict"
    )

    scores = cross_val_predict(
        pipeline, X, y,
        cv=cv, groups=groups,
        method="decision_function"
    )

    fold_accuracies = []

    for train_idx, test_idx in cv.split(X, y, groups):
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
        fold_pred = fold_model.predict(X.iloc[test_idx])
        fold_accuracies.append(
            accuracy_score(y[test_idx], fold_pred)
        )

    result = {
        "feature_set": model_name,
        "feature_count": len(columns),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "fold_accuracy_mean": np.mean(fold_accuracies),
        "fold_accuracy_std": np.std(fold_accuracies)
    }

    results.append(result)

    print(f"\n{model_name}")
    print("-" * 45)
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1-score:  {result['f1']:.4f}")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}")
    print(
        f"Fold accuracy: "
        f"{result['fold_accuracy_mean']:.4f} "
        f"+/- {result['fold_accuracy_std']:.4f}"
    )

results_df = pd.DataFrame(results)
output_path = os.path.join(
    OUTPUT_DIR, "feature_ablation_results.csv"
)
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("FEATURE COMPARISON SUMMARY")
print("=" * 65)
print(results_df.to_string(index=False))
print("\nSaved:", output_path)
print("\nExperiment complete.")
