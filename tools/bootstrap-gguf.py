#!/usr/bin/env python3
"""Install the pinned pure-Python GGUF wheel into the project without changing a base image."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

VERSION = "0.19.0"
ROOT = Path(__file__).resolve().parents[1]
metadata = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/gguf/{VERSION}/json", timeout=30))
wheel = next(item for item in metadata["urls"] if item["filename"].endswith("-py3-none-any.whl"))
with urllib.request.urlopen(wheel["url"], timeout=60) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
    raise ValueError("GGUF wheel checksum mismatch")
target = ROOT / ".python-deps"
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for name in archive.namelist():
        if not (target / name).resolve().is_relative_to(target.resolve()):
            raise ValueError("Unsafe wheel archive path")
    archive.extractall(target)
print(f"Installed gguf {VERSION} into {target}")
