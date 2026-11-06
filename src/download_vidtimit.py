import urllib.request
import zipfile
import time
from pathlib import Path

identities = [
    "fadg0", "faks0", "fcft0", "fcmh0",
    "fdac1", "fdrd1", "fedw0", "felc0",
    "fjas0", "fjem0", "fjre0", "fjwb0",
    "fkms0", "fram1", "mccs0", "mcem0",
    "mdab0", "mdbb0", "mdld0", "mgwt0",
    "mjar0", "mjsw0", "mmdb1", "mmdm2",
    "mpdf0", "mpgl0", "mrcz0", "mrgg0",
    "mrjo0", "msjs1", "mstk0", "mwbt0"
]

destination = Path("dataset/VidTIMIT")
destination.mkdir(parents=True, exist_ok=True)

success = []
failed = []

for i, identity in enumerate(identities, 1):
    filename = identity + ".zip"
    output = destination / filename
    temporary = destination / (filename + ".part")

    if output.exists() and zipfile.is_zipfile(output):
        print(f"[{i}/32] Already valid: {filename}")
        success.append(filename)
        continue

    url = f"https://zenodo.org/record/158963/files/{filename}"

    print(f"\n[{i}/32] Downloading {filename}")

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/octet-stream"
            }
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(temporary, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        print(
                            f"\rDownloaded {downloaded / 1048576:.1f} / "
                            f"{total / 1048576:.1f} MB",
                            end="",
                            flush=True
                        )

        if not zipfile.is_zipfile(temporary):
            raise ValueError("Invalid ZIP file received")

        temporary.replace(output)
        success.append(filename)
        print(f"\nSUCCESS: {filename}")

    except Exception as error:
        failed.append(filename)
        print(f"\nFAILED: {filename}: {error}")

        if temporary.exists():
            temporary.unlink()

    # Download only one file at a time.
    time.sleep(3)

print("\n========== FINAL SUMMARY ==========")
print("Successful:", len(success))
print("Failed:", len(failed))

if failed:
    print("Failed files:")
    for name in failed:
        print(name)

print("Saved in:", destination.resolve())
