"""
Bulk HDIC-only enrollment script — extracted directly from admin_api.py /enroll logic.
No hdic.adapter or hdic.db required.

Enrolls all persons from a dataset root folder into:
  db/watchlist_hdic.jsonl

Run from repo root:
    python tools/bulk_enroll_hdic.py --root dataset/test
    python tools/bulk_enroll_hdic.py --root dataset/test --max_imgs 80
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from preprocess.align           import load_and_align
from hdic.feature_extractor     import generate_embedding2
from hdic.encode_hv             import encode_embedding_to_hv
from hdic.cluster_enroll        import build_cluster_prototypes

# ── DB paths (same as admin_api.py) ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR    = REPO_ROOT / "db"
HDIC_FILE = DB_DIR / "watchlist_hdic.jsonl"

# ── JSONL helpers (copied from admin_api.py) ──────────────────────────────────
def load_jsonl(path: Path) -> list:
    out = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(json.loads(s))
    return out

def save_jsonl(path: Path, rows: list):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, str(path))

def find_person(rows: list, pid: str):
    for r in rows:
        if r.get("person_id") == pid:
            return r
    return None

def ensure_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if not HDIC_FILE.exists():
        HDIC_FILE.write_text("")

# ── Enroll one person (same logic as admin_api.py /enroll) ───────────────────
def enroll_person(person_id: str, image_paths: list, num_clusters: int = 3):
    """
    Mirrors admin_api.py /enroll exactly:
      1. load_and_align()
      2. generate_embedding2()       → 512-D FaceNet embedding
      3. encode_embedding_to_hv()    → 10,000-D binary HV
      4. build_cluster_prototypes()  → KMeans cluster prototypes
      5. Save to db/watchlist_hdic.jsonl
    """
    hd   = load_jsonl(HDIC_FILE)
    hdr  = find_person(hd, person_id)
    if hdr is None:
        hdr = {"person_id": person_id, "name": person_id, "prototypes": {}}
        hd.append(hdr)

    embeddings = []
    added, skipped = 0, 0

    for img_path in tqdm(image_paths, desc=f"  {person_id}", unit="img", leave=False):
        face = load_and_align(img_path)
        if face is None:
            skipped += 1
            continue
        try:
            emb = generate_embedding2(face)          # (512,) float32
            embeddings.append(emb)
            added += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(img_path)}: {e}")
            skipped += 1

    if len(embeddings) == 0:
        print(f"  [ERROR] No valid faces found for {person_id} — skipping.")
        return False

    # HDIC clustering (same as admin_api.py)
    hvs = [encode_embedding_to_hv(e) for e in embeddings]
    try:
        prototypes = build_cluster_prototypes(hvs, num_clusters=num_clusters)
        hdr["prototypes"] = {k: v.tolist() for k, v in prototypes.items()}
    except Exception as e:
        print(f"  [WARN] Clustering failed ({e}), using single prototype.")
        hdr["prototypes"] = {"cluster_0": hvs[0].tolist()}

    save_jsonl(HDIC_FILE, hd)

    n_clusters = len(hdr["prototypes"])
    print(f"  [OK] {person_id}  →  {added} images  →  {n_clusters} cluster(s)  "
          f"(skipped: {skipped})")
    return True

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bulk HDIC enrollment — writes to db/watchlist_hdic.jsonl"
    )
    parser.add_argument("--root",      required=True,
                        help="Dataset root folder  e.g. dataset/test")
    parser.add_argument("--max_imgs",  type=int, default=None,
                        help="Max images per person (default: all)")
    parser.add_argument("--clusters",  type=int, default=3,
                        help="Number of KMeans clusters (default: 3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear existing DB before enrolling")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] --root does not exist: {root}")
        return

    ensure_db()

    # Clear DB if requested
    if args.overwrite:
        HDIC_FILE.write_text("")
        print(f"[INFO] Cleared existing DB: {HDIC_FILE}")

    # Find all person folders
    persons = sorted([d for d in root.iterdir() if d.is_dir()])
    if not persons:
        print(f"[ERROR] No subfolders found in {root}")
        return

    print(f"\n[INFO] Found {len(persons)} person folder(s) in {root}")
    print(f"[INFO] Max images per person : {args.max_imgs or 'all'}")
    print(f"[INFO] KMeans clusters       : {args.clusters}")
    print(f"[INFO] Writing to            : {HDIC_FILE}\n")

    success, failed = 0, 0

    for person_dir in persons:
        pid = person_dir.name
        imgs = sorted([
            str(f) for f in person_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])
        if args.max_imgs:
            imgs = imgs[:args.max_imgs]

        if not imgs:
            print(f"  [SKIP] {pid} — no images found")
            failed += 1
            continue

        print(f"\n[{pid}]  {len(imgs)} image(s)")
        ok = enroll_person(pid, imgs, num_clusters=args.clusters)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"[DONE] Enrolled: {success}  |  Failed: {failed}")
    print(f"       DB saved → {HDIC_FILE}")


if __name__ == "__main__":
    main()