from pathlib import Path
import zipfile

source = Path("dataset/VidTIMIT")
destination = source / "extracted"
destination.mkdir(parents=True, exist_ok=True)

archives = sorted(source.glob("*.zip"))

print(f"Found {len(archives)} ZIP archives\n")

for index, archive in enumerate(archives, 1):
    identity = archive.stem
    output = destination / identity
    output.mkdir(parents=True, exist_ok=True)

    print(f"[{index}/{len(archives)}] Extracting {archive.name}...")

    with zipfile.ZipFile(archive, "r") as z:
        for member in z.infolist():
            target = (output / member.filename).resolve()

            if not target.is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")

        z.extractall(output)

print("\nExtraction completed!")

videos = [
    p for p in destination.rglob("*")
    if p.is_file() and p.suffix.lower() in [".avi", ".mov", ".mp4"]
]

print("Total extracted video files:", len(videos))

print("\nVideo count by identity:")
for identity_dir in sorted(destination.iterdir()):
    if identity_dir.is_dir():
        count = sum(
            1 for p in identity_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in [".avi", ".mov", ".mp4"]
        )
        print(f"{identity_dir.name}: {count}")

print("\nExtraction location:", destination.resolve())
