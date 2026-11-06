import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load predictions
df = pd.read_csv("outputs/feature_fusion/predictions.csv")

df["VideoID"] = df["VideoID"].astype(str)

models = df["Experiment"].unique().tolist()
baseline = "Eigenvalues only"

pair_ids = df["VideoID"].unique()

print("=" * 60)
print("PAIRED BOOTSTRAP ANALYSIS")
print("=" * 60)

# Validate each model independently
for model in models:
    subset = df[df["Experiment"] == model]

    if len(subset) != 98:
        raise ValueError(f"{model}: expected 98 rows, got {len(subset)}")

    for pair_id, pair in subset.groupby("VideoID"):
        labels = set(pair["Actual"].astype(int))

        if len(pair) != 2 or labels != {0, 1}:
            raise ValueError(
                f"Invalid matched pair: model={model}, ID={pair_id}"
            )

print("Validation successful: 49 matched real/fake pairs per model.")
print("Models:", len(models))
print("Bootstrap repetitions: 5000")

# Metric function
def metrics(data):
    y = data["Actual"].to_numpy(dtype=int)
    pred = data["Prediction"].to_numpy(dtype=int)
    score = data["Score"].to_numpy(dtype=float)

    return {
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, score)
    }

# Original results
original = {}

print("\nOriginal OOF metrics")
print("-" * 60)

for model in models:
    subset = df[df["Experiment"] == model]
    original[model] = metrics(subset)

    print(model, original[model])

# Build pair lookup for every model
pair_data = {}

for model in models:
    subset = df[df["Experiment"] == model]

    pair_data[model] = {
        pair_id: group.copy()
        for pair_id, group in subset.groupby("VideoID")
    }

# Bootstrap
rng = np.random.default_rng(42)
B = 5000

bootstrap = {
    model: {m: [] for m in ["Accuracy", "F1", "ROC_AUC"]}
    for model in models
}

differences = {
    model: {m: [] for m in ["Accuracy", "F1", "ROC_AUC"]}
    for model in models if model != baseline
}

for b in range(B):

    sampled_ids = rng.choice(
        pair_ids,
        size=len(pair_ids),
        replace=True
    )

    current = {}

    for model in models:

        pieces = [
            pair_data[model][pid]
            for pid in sampled_ids
        ]

        sample = pd.concat(pieces, ignore_index=True)
        current[model] = metrics(sample)

        for metric, value in current[model].items():
            bootstrap[model][metric].append(value)

    # Paired model differences on identical resampled pairs
    for model in models:
        if model == baseline:
            continue

        for metric in ["Accuracy", "F1", "ROC_AUC"]:
            differences[model][metric].append(
                current[model][metric] - current[baseline][metric]
            )

# Confidence intervals
summary_rows = []

print("\n95% bootstrap confidence intervals")
print("-" * 60)

for model in models:
    for metric in ["Accuracy", "F1", "ROC_AUC"]:

        values = np.asarray(bootstrap[model][metric])
        low, high = np.percentile(values, [2.5, 97.5])

        summary_rows.append({
            "Model": model,
            "Metric": metric,
            "Observed": original[model][metric],
            "CI_Lower": low,
            "CI_Upper": high
        })

        print(
            f"{model} | {metric}: "
            f"{original[model][metric]:.4f} "
            f"[{low:.4f}, {high:.4f}]"
        )

# Difference confidence intervals
difference_rows = []

print("\nPaired differences against Eigenvalues only")
print("-" * 60)

for model in models:
    if model == baseline:
        continue

    for metric in ["Accuracy", "F1", "ROC_AUC"]:

        values = np.asarray(differences[model][metric])
        low, high = np.percentile(values, [2.5, 97.5])

        observed = (
            original[model][metric]
            - original[baseline][metric]
        )

        difference_rows.append({
            "Model": model,
            "Metric": metric,
            "Difference": observed,
            "CI_Lower": low,
            "CI_Upper": high
        })

        print(
            f"{model} | {metric}: "
            f"{observed:+.4f} [{low:+.4f}, {high:+.4f}]"
        )

# Save outputs
os.makedirs("outputs/feature_fusion", exist_ok=True)

pd.DataFrame(summary_rows).to_csv(
    "outputs/feature_fusion/bootstrap_confidence_intervals.csv",
    index=False
)

pd.DataFrame(difference_rows).to_csv(
    "outputs/feature_fusion/paired_model_differences.csv",
    index=False
)

print("\nSaved results:")
print("outputs/feature_fusion/bootstrap_confidence_intervals.csv")
print("outputs/feature_fusion/paired_model_differences.csv")

print("\nBootstrap analysis completed.")
