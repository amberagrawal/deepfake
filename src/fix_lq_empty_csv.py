from pathlib import Path

p = Path("src/extract_deepfaketimit_lq.py")
s = p.read_text(encoding="utf-8")

old = '''if FAILURE_OUTPUT.exists():
    existing_failures = pd.read_csv(FAILURE_OUTPUT)
    if not existing_failures.empty:
        failure_rows = existing_failures.to_dict("records")'''

new = '''if FAILURE_OUTPUT.exists() and FAILURE_OUTPUT.stat().st_size > 0:
    try:
        existing_failures = pd.read_csv(FAILURE_OUTPUT)
        if not existing_failures.empty:
            failure_rows = existing_failures.to_dict("records")
    except pd.errors.EmptyDataError:
        failure_rows = []'''

if old not in s:
    raise SystemExit("Expected code not found. No changes made.")

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")

print("Empty failure CSV handling fixed.")
