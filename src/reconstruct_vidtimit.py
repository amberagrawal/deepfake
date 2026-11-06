from pathlib import Path
from PIL import Image
import cv2
import numpy as np

SOURCE = Path("dataset/VidTIMIT/extracted")
OUTPUT = Path("dataset/VidTIMIT/reconstructed_videos")
FPS = 25

OUTPUT.mkdir(parents=True, exist_ok=True)

identities = sorted(p for p in SOURCE.iterdir() if p.is_dir())

total_videos = 0
failed = []

print("========== RECONSTRUCTING VIDTIMIT VIDEOS ==========")

for identity_dir in identities:
    identity = identity_dir.name
    video_root = identity_dir / identity / "video"

    if not video_root.exists():
        print(f"SKIPPED: {identity} - video folder not found")
        continue

    sequences = sorted(p for p in video_root.iterdir() if p.is_dir())

    for sequence in sequences:
        frames = sorted(
            [p for p in sequence.iterdir() if p.is_file()],
            key=lambda p: int(p.name) if p.name.isdigit() else 10**12
        )

        if not frames or not all(p.name.isdigit() for p in frames):
            continue

        destination_dir = OUTPUT / identity
        destination_dir.mkdir(parents=True, exist_ok=True)

        output_file = destination_dir / f"{sequence.name}.mp4"
        writer = None

        try:
            for index, frame_path in enumerate(frames):
                with Image.open(frame_path) as image:
                    image = image.convert("RGB")
                    frame = cv2.cvtColor(
                        np.array(image),
                        cv2.COLOR_RGB2BGR
                    )

                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_file),
                        fourcc,
                        FPS,
                        (width, height)
                    )

                    if not writer.isOpened():
                        raise RuntimeError("Could not open video writer")

                writer.write(frame)

            if writer is not None:
                writer.release()
                writer = None

            if not output_file.exists() or output_file.stat().st_size == 0:
                raise RuntimeError("Output video was not created")

            total_videos += 1
            print(
                f"SUCCESS: {identity}/{sequence.name} "
                f"({len(frames)} frames)"
            )

        except Exception as error:
            failed.append((identity, sequence.name, str(error)))
            print(f"FAILED: {identity}/{sequence.name}: {error}")

        finally:
            if writer is not None:
                writer.release()

print("\n========== RECONSTRUCTION SUMMARY ==========")
print("Identities inspected:", len(identities))
print("Videos reconstructed:", total_videos)
print("Failed sequences:", len(failed))

for item in failed:
    print("FAILED:", item)

print("Output folder:", OUTPUT.resolve())
