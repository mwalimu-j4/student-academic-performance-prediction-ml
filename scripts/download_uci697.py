"""Download the official UCI 697 CSV archive."""

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
ROOT = Path(__file__).resolve().parents[1]

with urlopen(URL, timeout=120) as response:
    archive = response.read()
raw = ROOT / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)
zip_path = raw / "uci-697.zip"
zip_path.write_bytes(archive)
with ZipFile(zip_path) as zf:
    zf.extractall(raw / "uci-697")
zip_path.unlink()
print(raw / "uci-697" / "data.csv")
