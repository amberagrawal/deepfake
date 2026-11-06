from pathlib import Path

p = Path("src/extract_deepfaketimit_lq.py")
s = p.read_text(encoding="utf-8")

old = '''successful_rows = []
failure_rows = []

for index, job in enumerate(jobs, start=1):'''

new = '''# Resume support: retain previously successful videos and skip them.
successful_rows = []
failure_rows = []

if FEATURE_OUTPUT.exists():
    existing = pd.read_csv(FEATURE_OUTPUT)
    if not existing.empty:
        successful_rows = existing.to_dict("records")

if FAILURE_OUTPUT.exists():
    existing_failures = pd.read_csv(FAILURE_OUTPUT)
    if not existing_failures.empty:
        failure_rows = existing_failures.to_dict("records")

completed_paths = {
    str(row["video_path"]).lower()
    for row in successful_rows
}

print("Previously successful videos:", len(completed_paths))
print("Previously recorded failures:", len(failure_rows))

remaining_jobs = [
    job for job in jobs
    if str(Path(job["video_path"])).lower() not in completed_paths
]

print("Videos remaining to process:", len(remaining_jobs))

for index, job in enumerate(remaining_jobs, start=1):'''

if old not in s:
    raise SystemExit(
        "Could not find the exact processing-loop initialization. "
        "No changes were made."
    )

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")

print("Resume support added successfully.")
