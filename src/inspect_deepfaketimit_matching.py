import pandas as pd
from pathlib import Path

path = Path("outputs/dataset/deepfaketimit_hq_matching.csv")
df = pd.read_csv(path)

print("Total rows:", len(df))
print("Unique utterance names:", df["Utterance"].nunique())

print("\nAll unique utterance names:")
for name in sorted(df["Utterance"].unique()):
    print(name)

print("\nNumber of videos per target identity:")
print(df.groupby("TargetIdentity").size().to_string())

print("\nSample original/fake filenames:")
for _, row in df.head(15).iterrows():
    print("REAL:", Path(row["RealVideoPath"]).name)
    print("FAKE:", Path(row["FakeVideoPath"]).name)
    print()
