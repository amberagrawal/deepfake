import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

folder = "outputs/eigen_jitter_threshold"

df = pd.read_csv(f"{folder}/video_jitter_scores.csv")

pairs = df.pivot(
    index="video_id",
    columns="label",
    values="jitter_mean"
).dropna()

real = pairs["real"].to_numpy()
fake = pairs["fake"].to_numpy()

# Difference: fake minus real
diff = fake - real

# Paired statistical test
result = wilcoxon(
    fake,
    real,
    alternative="two-sided",
    method="auto"
)

# Bootstrap confidence interval for median difference
rng = np.random.default_rng(42)
bootstrap_medians = []

for _ in range(10000):
    sample = rng.choice(diff, size=len(diff), replace=True)
    bootstrap_medians.append(np.median(sample))

ci_low, ci_high = np.percentile(
    bootstrap_medians, [2.5, 97.5]
)

# Effect size: paired standardized mean difference
effect_size = np.mean(diff) / np.std(diff, ddof=1)

print("=" * 55)
print("PAIRED EIGEN-JITTER STATISTICAL ANALYSIS")
print("=" * 55)

print("Number of matched pairs:", len(diff))
print("Real mean jitter:", np.mean(real))
print("Fake mean jitter:", np.mean(fake))
print("Mean paired difference:", np.mean(diff))
print("Median paired difference:", np.median(diff))

print("\nDirection:")
print("Fake lower:", np.sum(diff < 0))
print("Fake higher:", np.sum(diff > 0))
print("Equal:", np.sum(diff == 0))

print("\nWilcoxon signed-rank test:")
print("Statistic:", result.statistic)
print("P-value:", result.pvalue)

print("\n95% bootstrap CI for median difference:")
print("Lower:", ci_low)
print("Upper:", ci_high)

print("\nPaired standardized effect size (Cohen's dz):")
print(effect_size)

print("\nInterpretation:")
if result.pvalue < 0.05:
    print("The paired difference is statistically significant at 0.05.")
else:
    print("The paired difference is NOT statistically significant at 0.05.")

if ci_high < 0:
    print("The bootstrap interval supports lower jitter in fake videos.")
elif ci_low > 0:
    print("The bootstrap interval supports higher jitter in fake videos.")
else:
    print("The bootstrap interval includes zero.")

print("=" * 55)
