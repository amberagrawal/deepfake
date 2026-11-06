from pathlib import Path
import pandas as pd

root = Path("dataset/DeepfakeTIMIT")
extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}

files = [
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

rows = []

for p in files:
    relative = p.relative_to(root)
    parts = relative.parts

    quality = parts[0] if len(parts) >= 2 else "OUTSIDE_QUALITY_FOLDER"
    identity = parts[1] if len(parts) >= 3 else "ROOT"

    rows.append({
        "Quality": quality,
        "Identity": identity,
        "Filename": p.name,
        "Extension": p.suffix.lower(),
        "RelativePath": str(relative)
    })

df = pd.DataFrame(rows)

print("\n========== TOTAL VIDEO COUNT ==========")
print(len(df))

print("\n========== COUNT BY QUALITY ==========")
print(df.groupby("Quality").size().to_string())

print("\n========== COUNT BY EXTENSION ==========")
print(df.groupby("Extension").size().to_string())

print("\n========== COUNT BY IDENTITY AND QUALITY ==========")
print(
    df.groupby(["Quality", "Identity"])
      .size()
      .to_string()
)

print("\n========== FILES OUTSIDE QUALITY FOLDERS ==========")
print(
    df[df["Quality"] == "OUTSIDE_QUALITY_FOLDER"]
      ["RelativePath"].to_string(index=False)
)

print("\n========== FIRST 30 VIDEO PATHS ==========")
print(df["RelativePath"].head(30).to_string(index=False))

output = Path("outputs/dataset")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(output / "dee f aketimit_inventory.csv", index=False)

print("\nInventory saved in outputs/dataset/")
