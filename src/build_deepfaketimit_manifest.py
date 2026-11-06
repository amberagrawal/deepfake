from pathlib import Path
import re
import csv

ROOT = Path(__file__).resolve().parents[1]

FAKE_ROOTS = {
    "HQ": ROOT / "dataset" / "DeepfakeTIMIT" / "higher_quality",
    "LQ": ROOT / "dataset" / "DeepfakeTIMIT" / "lower_quality",
}

REAL_ROOT = ROOT / "dataset" / "VidTIMIT" / "reconstructed_videos"
OUT = ROOT / "outputs" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"^(?P<utterance>.+)-video-(?P<swapped_identity>[^.]+)\.avi$", re.I)

for quality, fake_root in FAKE_ROOTS.items():
    rows = []
    unmatched = []
    fake_count = 0

    for fake_path in sorted(fake_root.rglob("*.avi")):
        fake_count += 1
        target_identity = fake_path.parent.name
        match = pattern.match(fake_path.name)

        if not match:
            unmatched.append((str(fake_path), "Filename did not match expected pattern"))
            continue

        utterance = match.group("utterance")
        swapped_identity = match.group("swapped_identity")

        real_path = REAL_ROOT / target_identity / f"{utterance}.mp4"

        if not real_path.exists():
            unmatched.append((str(fake_path), f"Missing real counterpart: {real_path}"))
            continue

        rows.append({
            "Quality": quality,
            "Label": "fake",
            "TargetIdentity": target_identity,
            "SwappedIdentity": swapped_identity,
            "Utterance": utterance,
            "RealVideoPath": str(real_path),
            "FakeVideoPath": str(fake_path.resolve()),
        })

    output_file = OUT / f"dee pf aketimit_{quality.lower()}_matching.csv"
    # Use a simple filename without spaces.
    output_file = OUT / f"deepfaketimit_{quality.lower()}_matching.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Quality", "Label", "TargetIdentity", "SwappedIdentity",
            "Utterance", "RealVideoPath", "FakeVideoPath"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n========== {quality} MATCHING ==========")
    print("Fake videos found:", fake_count)
    print("Valid real/fake pairs:", len(rows))
    print("Unmatched videos:", len(unmatched))
    print("Unique target identities:", len(set(r["TargetIdentity"] for r in rows)))
    print("Unique utterances:", len(set(r["Utterance"] for r in rows)))
    print("Manifest:", output_file)

    if rows:
        print("\nFirst 5 matches:")
        for row in rows[:5]:
            print(
                f"Target={row['TargetIdentity']} | "
                f"Utterance={row['Utterance']} | "
                f"Swapped identity={row['SwappedIdentity']}"
            )

    if unmatched:
        print("\nFirst 10 unmatched:")
        for item, reason in unmatched[:10]:
            print(item)
            print("  Reason:", reason)

    if len(rows) != fake_count:
        print("\nWARNING: Not every fake video has a matching original.")
