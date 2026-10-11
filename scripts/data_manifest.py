#!/usr/bin/env python3
"""Checksum manifest for every input file (spec/data-manifest.json). Standard library only.

    python3 scripts/data_manifest.py --write              hash the files present, merge into the manifest
    python3 scripts/data_manifest.py --verify             full sha256 of every file the manifest lists
    python3 scripts/data_manifest.py --verify --quick     presence + size only (what the launcher uses)
    python3 scripts/data_manifest.py --verify --quick --paths data/code15/exams_part0.hdf5 ...
    python3 scripts/data_manifest.py --manifest-sha256    print the manifest's own sha256
    python3 scripts/data_manifest.py --published-md5 PATH print Zenodo's md5 of the zip PATH unpacks from

--write only adds or refreshes entries for files that exist on this machine; it never drops
entries. Run it locally for part0 and the CSVs, then on the pod for parts 1-17.
--verify exits 1 on any mismatch, unrecorded file or (without --allow-missing) missing file.

Provenance: the code15 / samitrop `published_*` fields were copied from Zenodo's records API on
2026-10-11. Zenodo publishes md5 for the zips and for the CSVs, not for the unzipped HDF5, so
for the HDF5 the check is: download the zip, compare its md5, unzip, compare the sha256 here.
PTB-XL ships its own SHA256SUMS.txt; --verify checks every file against it.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "spec" / "data-manifest.json"
CHUNK = 8 * 1024 * 1024
ZENODO_RETRIEVED = "2026-10-11"

DATASETS = {
    "code15": {"name": "CODE-15%", "doi": "10.5281/zenodo.4916206", "version": "1.0.0",
               "url": "https://zenodo.org/records/4916206"},
    "code15_labels": {"name": "CODE-15% Chagas labels (PhysioNet/CinC Challenge 2025)", "doi": None,
                      "version": None,
                      "url": "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"},
    "samitrop": {"name": "SaMi-Trop", "doi": "10.5281/zenodo.4905618", "version": "1.0.0",
                 "url": "https://zenodo.org/records/4905618"},
    "ptbxl": {"name": "PTB-XL", "doi": "10.13026/kfzx-aw45", "version": "1.0.3",
              "url": "https://physionet.org/content/ptb-xl/1.0.3/"},
}

# (zip size, zip md5) as published by Zenodo for record 4916206.
CODE15_ZIPS = {
    0: (2672786404, "2bed0dc753d16beef8c2f7627e2b6ea4"), 1: (2668177892, "b32446cdb93247d07550509a204a061d"),
    2: (2675394813, "e2862a75eeb6245b148c6c520245c0e0"), 3: (2676107654, "623e073c1e1323cb69a219e5f9bedaf3"),
    4: (2673870870, "e5f958b7c31bd82a6bc76fcf6aed5713"), 5: (2673013182, "2730cb080d03d61681ded6609c18c9d8"),
    6: (2667377774, "4cd7537330f0a62ffcc6cc40c63774eb"), 7: (2671653394, "aec8d97e8dde9ac15f5c671c168fdfbe"),
    8: (2666112083, "96adb16ee032e4e41528c04fd7b582ee"), 9: (2668370881, "8d74d7032396298304b51e447b73d40f"),
    10: (2672404916, "26bf9e387289dabbb140c0453853872b"), 11: (2665653597, "dd99137b6c199c9558bc2c2b6ae0e4dc"),
    12: (2674194718, "c25b38260fb46edff089fa56eb442ddb"), 13: (2667409911, "5472ba3186e39bd03e888e486143bf7a"),
    14: (2671739754, "0a0458779f5e795df20cb07db6e50682"), 15: (2695359122, "cf99b9c54cf7c15b9b683511cd5e6d5a"),
    16: (2725249232, "64c210e19cf1c0abac3e643f88708ebd"), 17: (782633664, "6e9ac2e36197c4d301df91d7dc6877c0"),
}


def _known_files() -> dict:
    """path -> static facts about where each input file comes from."""
    z15 = "https://zenodo.org/records/4916206/files/"
    known = {}
    for i, (zsize, zmd5) in CODE15_ZIPS.items():
        known[f"data/code15/exams_part{i}.hdf5"] = {
            "dataset": "code15", "required": True,
            "source_url": f"{z15}exams_part{i}.zip", "published_zip_size": zsize, "published_zip_md5": zmd5}
    known["data/code15/exams.csv"] = {
        "dataset": "code15", "required": True, "source_url": f"{z15}exams.csv",
        "published_size": 35500900, "published_md5": "0107516d3f63864498fb77d15799cc95"}
    known["data/code15/code15_chagas_labels.csv"] = {
        "dataset": "code15_labels", "required": True, "source_url": DATASETS["code15_labels"]["url"],
        "note": "extracted from code15_chagas_labels.zip; no published checksum"}
    known["data/code15/code15_chagas_labels.zip"] = {
        "dataset": "code15_labels", "required": False, "source_url": DATASETS["code15_labels"]["url"]}
    z_st = "https://zenodo.org/records/4905618/files/"
    known["data/samitrop/exams.hdf5"] = {
        "dataset": "samitrop", "required": True, "source_url": f"{z_st}exams.zip",
        "published_zip_size": 264538267, "published_zip_md5": "a7b65f115b0222ad2ecbc6a422496fdd"}
    known["data/samitrop/exams.csv"] = {
        "dataset": "samitrop", "required": True, "source_url": f"{z_st}exams.csv",
        "published_size": 87713, "published_md5": "6c9007a0427f7c3d9e1b6fb091231a67"}
    return known


KNOWN = _known_files()
PTBXL_ROOT = "data/ptbxl"
PTBXL_SUMS = f"{PTBXL_ROOT}/SHA256SUMS.txt"


def digest(path: Path, md5: bool = False) -> tuple:
    h, m = hashlib.sha256(), hashlib.md5() if md5 else None
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            if m:
                m.update(b)
    return h.hexdigest(), (m.hexdigest() if m else None)


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"schema_version": 1, "datasets": DATASETS, "zenodo_checksums_retrieved": ZENODO_RETRIEVED,
            "files": {}, "ptbxl": {}}


def save_manifest(m: dict) -> None:
    m["datasets"], m["zenodo_checksums_retrieved"] = DATASETS, ZENODO_RETRIEVED
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=MANIFEST.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, MANIFEST)


def ptbxl_listed(root: Path) -> list:
    """(sha256, relative path) for every line of PTB-XL's own SHA256SUMS.txt."""
    out = []
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out.append((h, rel.strip().lstrip("*")))
    return out


def do_write(args) -> int:
    m = load_manifest()
    files, bad = m.setdefault("files", {}), 0
    todo = [p for p in KNOWN if (ROOT / p).is_file()]
    for p in todo:
        size = (ROOT / p).stat().st_size
        old = files.get(p)
        if old and old.get("size") == size and not args.rehash:
            print(f"keep   {p}")
            continue
        sha, md5 = digest(ROOT / p, md5="published_md5" in KNOWN[p])
        entry = {"size": size, "sha256": sha, **KNOWN[p]}
        if md5 is not None:
            entry["md5"] = md5
            entry["md5_matches_published"] = md5 == KNOWN[p]["published_md5"] and size == KNOWN[p]["published_size"]
            if not entry["md5_matches_published"]:
                print(f"MISMATCH vs Zenodo published md5/size: {p}", file=sys.stderr)
                bad += 1
        files[p] = entry
        print(f"hashed {p}  {size} bytes  {sha[:16]}")
    root = ROOT / PTBXL_ROOT
    if (root / "SHA256SUMS.txt").is_file():
        listed = ptbxl_listed(root)
        m["ptbxl"] = {"dataset": "ptbxl", "root": PTBXL_ROOT, "sums_file": "SHA256SUMS.txt",
                      "sums_file_sha256": digest(root / "SHA256SUMS.txt")[0], "n_files": len(listed),
                      "source_url": DATASETS["ptbxl"]["url"],
                      "note": "every listed file is checked against the dataset's own SHA256SUMS.txt by --verify"}
        print(f"ptbxl  SHA256SUMS.txt lists {len(listed)} files")
    save_manifest(m)
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(files)} files recorded")
    return 1 if bad else 0


def check_file(p: str, entry: dict, quick: bool) -> str:
    """Return '' if the file matches its manifest entry, else the reason it does not."""
    fp = ROOT / p
    if not fp.is_file():
        return "missing"
    if fp.stat().st_size != entry["size"]:
        return f"size {fp.stat().st_size} != recorded {entry['size']}"
    if not quick and digest(fp)[0] != entry["sha256"]:
        return "sha256 differs from the manifest"
    return ""


def check_ptbxl(m: dict, quick: bool, jobs: int, allow_missing: bool) -> list:
    root, rec = ROOT / PTBXL_ROOT, m.get("ptbxl") or {}
    if not rec:
        print("ptbxl: not in manifest, skipped")
        return []
    if not (root / "SHA256SUMS.txt").is_file():
        if allow_missing:
            print("ptbxl: not on this machine, skipped")
            return []
        return ["ptbxl: SHA256SUMS.txt missing"]
    fails = []
    if digest(root / "SHA256SUMS.txt")[0] != rec["sums_file_sha256"]:
        fails.append("ptbxl: SHA256SUMS.txt differs from the manifest")
    listed = ptbxl_listed(root)
    if len(listed) != rec["n_files"]:
        fails.append(f"ptbxl: SHA256SUMS.txt lists {len(listed)} files, manifest recorded {rec['n_files']}")

    def one(item):
        want, rel = item
        fp = root / rel
        if not fp.is_file():
            return f"ptbxl: {rel} missing"
        if not quick and digest(fp)[0] != want:
            return f"ptbxl: {rel} sha256 differs from SHA256SUMS.txt"
        return ""

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(one, listed), 1):
            if r:
                fails.append(r)
            if i % 10000 == 0:
                print(f"ptbxl: {i}/{len(listed)} checked", flush=True)
    print(f"ptbxl: {len(listed)} files, {len([f for f in fails if f.startswith('ptbxl: ') and 'SHA256SUMS.txt' not in f])} bad")
    return fails


def do_verify(args) -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST.relative_to(ROOT)}; run --write first", file=sys.stderr)
        return 2
    m = load_manifest()
    files, fails = m["files"], []
    if args.paths:
        want = [p[2:] if p.startswith("./") else p for p in args.paths]
        in_ptbxl = [p for p in want if p == PTBXL_ROOT or p.startswith(PTBXL_ROOT + "/")]
        for p in want:
            if p in in_ptbxl:
                continue
            if p not in files:
                fails.append(f"{p}: not in the manifest")
                continue
            why = check_file(p, files[p], quick=args.quick)
            print(("OK    " if not why else "FAIL  ") + p + (f"  ({why})" if why else ""))
            if why:
                fails.append(f"{p}: {why}")
        if in_ptbxl:
            fails += check_ptbxl(m, True, args.jobs, False)
    else:
        names = sorted(set(files) | {p for p in KNOWN if (ROOT / p).is_file()})

        def one(p):
            if p not in files:
                return p, "on disk but not in the manifest (run --write)"
            if not (ROOT / p).is_file():
                if args.allow_missing or not files[p].get("required", True):
                    return p, None
                return p, "missing"
            return p, check_file(p, files[p], args.quick)

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for p, why in ex.map(one, names):
                if why is None:
                    print(f"SKIP  {p}  (not on this machine)")
                else:
                    print(("OK    " if not why else "FAIL  ") + p + (f"  ({why})" if why else ""), flush=True)
                    if why:
                        fails.append(f"{p}: {why}")
        fails += check_ptbxl(m, args.quick, args.jobs, args.allow_missing)
    mode = "size/presence only" if args.quick else "full sha256"
    print(f"\nverify ({mode}): {'FAILED' if fails else 'PASSED'}  manifest sha256 {digest(MANIFEST)[0]}")
    for f in fails[:25]:
        print(f"  - {f}", file=sys.stderr)
    if len(fails) > 25:
        print(f"  ... and {len(fails) - 25} more", file=sys.stderr)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--manifest-sha256", action="store_true")
    g.add_argument("--published-md5", metavar="PATH", help="print the published md5 of the source zip/CSV")
    ap.add_argument("--rehash", action="store_true", help="--write: rehash files whose size is unchanged")
    ap.add_argument("--quick", action="store_true", help="--verify: presence and size only")
    ap.add_argument("--allow-missing", action="store_true", help="--verify: files not on this machine are skipped")
    ap.add_argument("--paths", nargs="+", help="--verify: check only these paths (repo-relative)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.manifest_sha256:
        print(digest(MANIFEST)[0])
        return 0
    if a.published_md5:
        k = KNOWN.get(a.published_md5, {})
        print(k.get("published_zip_md5") or k.get("published_md5") or "")
        return 0
    return do_write(a) if a.write else do_verify(a)


if __name__ == "__main__":
    sys.exit(main())
