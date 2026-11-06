import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("outputs/features/uadfv_features.csv")
df["video_id"] = df["video_id"].astype(str)
df["label"] = df["label"].astype(str).str.lower().str.strip()

y = (df["label"] == "fake").astype(int).values
groups = df["video_id"].values

eigen_cols = [
    c for c in df.columns if c.startswith("eigen_")
]

jitter_cols = [
    c for c in df.columns
    if c.startswith("jitter_")
    or c in ["overall_jitter_mean_abs", "overall_jitter_std"]
]

combined_cols = eigen_cols + jitter_cols

print("=" * 55)
print("NESTED FEATURE SELECTION AND FUSION EXPERIMENT")
print("=" * 55)
print("Videos:", len(df))
print("Matched video pairs:", len(np.unique(groups)))
print("Eigenvalue features:", len(eigen_cols))
print("Eigen-Jitter features:", len(jitter_cols))
print("Combined features:", len(combined_cols))

assert len(df) == 98
assert len(eigen_cols) == 76
assert len(jitter_cols) == 78
assert len(combined_cols) == 154

# --------------------------------------------------
# 2. Define four approaches
# --------------------------------------------------

experiments = {
    "Eigenvalues only": {
        "columns": eigen_cols,
        "selection": False
    },
    "Eigen-Jitter only": {
        "columns": jitter_cols,
        "selection": False
    },
    "All combined features": {
        "columns": combined_cols,
        "selection": False
    },
    "Selected combined features": {
        "columns": combined_cols,
        "selection": True
    }
}

# --------------------------------------------------
# 3. Outer cross-validation
# --------------------------------------------------

outer_cv = GroupKFold(n_splits=5)

all_results = []
all_predictions = []

for experiment_name, config in experiments.items():

    print("\n" + "=" * 55)
    print("EXPERIMENT:", experiment_name)
    print("=" * 55)

    X = df[config["columns"]].values

    predictions = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), 1
    ):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        # All preprocessing and feature selection are inside
        # the pipeline, so they are fitted only on training data.
        if config["selection"]:

            pipeline = Pipeline([
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced"
                ))
            ])

            param_grid = {
                "select__k": [10, 20, 40, "all"],
                "svm__C": [0.1, 1, 10]
            }

        else:

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced"
                ))
            ])

            param_grid = {
                "svm__C": [0.1, 1, 10]
            }

        inner_cv = GroupKFold(n_splits=4)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            refit=True,
            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train,
            groups=train_groups
        )

        best_model = search.best_estimator_

        predictions[test_idx] = best_model.predict(X_test)
        scores[test_idx] = best_model.decision_function(X_test)

        fold_accuracy = accuracy_score(
            y_test, predictions[test_idx]
        )

        print(
            f"Fold {fold}: "
            f"Accuracy={fold_accuracy:.4f}, "
            f"Best parameters={search.best_params_}"
        )

    # --------------------------------------------------
    # 4. Evaluate out-of-fold predictions
    # --------------------------------------------------

    accuracy = accuracy_score(y, predictions)
    balanced = balanced_accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, scores)

    cm = confusion_matrix(y, predictions)

    print("\nOUT-OF-FOLD RESULTS")
    print("-" * 40)
    print("Accuracy:", round(accuracy, 4))
    print("Balanced accuracy:", round(balanced, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix:")
    print(cm)

    all_results.append({
        "Experiment": experiment_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    for i in range(len(df)):
        all_predictions.append({
            "Experiment": experiment_name,
            "VideoID": groups[i],
            "Label": df.iloc[i]["label"],
            "Actual": y[i],
            "Prediction": predictions[i],
            "Score": scores[i],
            "Correct": predictions[i] == y[i]
        })

# --------------------------------------------------
# 5. Save and display comparison
# --------------------------------------------------

os.makedirs("outputs/feature_fusion", exist_ok=True)

results_df = pd.DataFrame(all_results)
predictions_df = pd.DataFrame(all_predictions)

results_df.to_csv(
    "outputs/feature_fusion/summary.csv",
    index=False
)

predictions_df.to_csv(
    "outputs/feature_fusion/predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL FEATURE FUSION COMPARISON")
print("=" * 65)

print(
    results_df.sort_values(
        "Accuracy", ascending=False
    ).to_string(index=False)
)

print("\nSaved:")
print("outputs/feature_fusion/summary.csv")
print("outputs/feature_fusion/predictions.csv")

