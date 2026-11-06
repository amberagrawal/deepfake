from pathlib import Path
from PIL import Image
import collections

root = Path("dataset/VidTIMIT/extracted/fadg0")

print("========== DIRECTORY STRUCTURE ==========")

for folder in sorted(root.rglob("*")):
    if folder.is_dir():
        files = [p for p in folder.iterdir() if p.is_file()]
        print(f"{folder.relative_to(root)} : {len(files)} files")

print("\n========== IMAGE FORMAT CHECK ==========")

image_folder = root / "fadg0" / "video" / "head"

images = sorted(
    [p for p in image_folder.iterdir() if p.is_file()]
)

print("Image count:", len(images))

for path in images[:10]:
    try:
        with Image.open(path) as img:
            print(
                path.name,
                "| Format:", img.format,
                "| Size:", img.size,
                "| Mode:", img.mode
            )
    except Exception as e:
        print(path.name, "| Error:", e)

print("\n========== IMAGE SEQUENCE CHECK ==========")

print("First frame:", images[0].name if images else "None")
print("Last frame:", images[-1].name if images else "None")
