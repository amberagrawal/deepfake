import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/dataset/deepfaketimit_hq_matching.csv")

print("Total fake videos:", len(df))
print("Unique real video paths:", df["RealVideoPath"].nunique())
print("Unique fake video paths:", df["FakeVideoPath"].nunique())

print("\nPairs per identity:")
counts = df.groupby("TargetIdentity").size()
print(counts.to_string())

print("\nMissing real files:",
      sum(not Path(p).exists() for p in df["RealVideoPath"]))

print("Missing fake files:",
      sum(not Path(p).exists() for p in df["FakeVideoPath"]))

print("\nIdentities with count other than 10:",
      counts[counts != 10].to_dict())

print("\nUnique real recordings:", df["RealVideoPath"].nunique())

if (
    len(df) == 320
    and df["RealVideoPath"].nunique() == 320
    and df["FakeVideoPath"].nunique() == 320
    and (counts == 10).all()
    and all(Path(p).exists() for p in df["RealVideoPath"])
    and all(Path(p).exists() for p in df["FakeVideoPath"])
):
    print("\nRESULT: PASS — 320 distinct matched pairs.")
else:
    print("\nRESULT: CHECK REQUIRED — integrity conditions not all satisfied.")
