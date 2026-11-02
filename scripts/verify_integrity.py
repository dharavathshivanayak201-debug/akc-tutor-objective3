"""Verify preserved public inputs; explicitly account for one withheld table."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / "provenance/preserved-files.json").read_text())
    exclusions = json.loads((ROOT / "provenance/release-exclusions.json").read_text())
    failures = []
    for row in records:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            failures.append(row["path"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append(row["path"])
    excluded = {r["path"] for r in exclusions}
    original = json.loads((ROOT / "research/MANIFEST_SHA256.json").read_text())
    for row in original:
        path = ROOT / "research" / row["path"]
        if row["path"] in excluded:
            if path.exists():
                failures.append("Withheld input unexpectedly published: " + row["path"])
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append("Supplied manifest: " + row["path"])
    if failures:
        raise SystemExit("Integrity failed:\n" + "\n".join(failures))
    print(f"Integrity passed: {len(records)} preserved public inputs; {len(exclusions)} explicitly withheld table.")


if __name__ == "__main__":
    main()
