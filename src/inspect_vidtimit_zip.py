import zipfile
from pathlib import Path

archive = Path("dataset/VidTIMIT/fadg0.zip")

with zipfile.ZipFile(archive) as z:
    files = [x for x in z.infolist() if not x.is_dir()]

    print("Archive:", archive.name)
    print("Total files:", len(files))
    print("Total uncompressed size:", sum(x.file_size for x in files))

    print("\nFirst 40 files inside archive:")
    for item in files[:40]:
        print(f"{item.filename}  ({item.file_size:,} bytes)")

    print("\nFile extensions:")
    extensions = {}
    for item in files:
        ext = Path(item.filename).suffix.lower() or "[no extension]"
        extensions[ext] = extensions.get(ext, 0) + 1

    for ext, count in sorted(extensions.items()):
        print(f"{ext}: {count}")
