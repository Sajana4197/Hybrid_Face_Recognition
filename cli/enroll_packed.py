# cli/enroll_packed.py
import argparse, glob, os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits           # -> (96,) {0,1}
from hdic.adapter import encode_hv                         # -> (10000,) {0,1}
from hdic.cluster_enroll import build_cluster_prototypes   # -> may return dict | list | tuple
from db.packed_store import PackedStore

DB_DIR = Path("db")  # adjust if you store DB elsewhere

# ---------------------------
# Helpers
# ---------------------------
def _expand(pattern: str):
    """Expand a folder or glob pattern into a sorted list of image paths."""
    if os.path.isdir(pattern):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            files.extend(glob.glob(os.path.join(pattern, ext)))
        return sorted(files)
    return sorted(glob.glob(pattern))

def _coerce_protos_to_matrix(protos, bits: int = 10000) -> np.ndarray:
    """
    Coerce build_cluster_prototypes() output into an array of shape (K, bits) uint8 in {0,1}.
    Accepts:
      - dict: {cluster_id: vector}
      - list/tuple of vectors
      - list/tuple of dicts with a vector under keys: 'proto','hv','vector','code','embedding','bits'
      - tuple like (protos, meta) -> use first element
    """
    # Unpack tuple returns like (protos, meta)
    if isinstance(protos, tuple) and len(protos) >= 1:
        protos = protos[0]

    def _extract_vec(x):
        if isinstance(x, dict):
            for k in ("proto", "hv", "vector", "code", "embedding", "bits"):
                if k in x:
                    return np.asarray(x[k])
            # if it's a dict-of-vectors at top level, handled outside
            # else fall through to asarray (will fail later with a clear error)
        return np.asarray(x)

    if isinstance(protos, dict):
        mats = [np.asarray(v) for v in protos.values()]
    elif isinstance(protos, (list, tuple)) and len(protos) > 0 and isinstance(protos[0], dict):
        mats = [_extract_vec(d) for d in protos]
    elif isinstance(protos, (list, tuple)):
        mats = [np.asarray(v) for v in protos]
    else:
        mats = [np.asarray(protos)]

    try:
        mat = np.stack(mats, axis=0)
    except Exception as e:
        raise ValueError(f"Could not stack prototypes into a matrix: {e}. "
                         f"Sample types: {[type(m).__name__ for m in mats[:3]]}")

    # Normalize dtype and ensure binary {0,1}
    if mat.dtype == bool:
        mat = mat.astype(np.uint8, copy=False)
    elif not np.issubdtype(mat.dtype, np.integer):
        # e.g., float in [0,1] or real scores — threshold at 0.5
        mat = (mat >= 0.5).astype(np.uint8)
    else:
        # integer but possibly not 0/1; clamp to {0,1}
        mat = (mat > 0).astype(np.uint8)

    if mat.ndim != 2 or mat.shape[1] != bits:
        raise ValueError(f"Prototypes must have shape (K,{bits}); got {mat.shape}")

    return mat

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Person ID")
    ap.add_argument("--name", required=True, help="Person name")
    ap.add_argument("--images", required=True, help="Folder or glob of images")
    ap.add_argument("--hdic_clusters", type=int, default=3)
    args = ap.parse_args()

    paths = _expand(args.images)
    if not paths:
        print(f"[ERROR] no images for {args.images}")
        return

    nh_rows, hv_rows = [], []
    used = skipped = 0

    t = tqdm(paths, unit="img", desc=f"Enroll {args.id}")
    for p in t:
        img = load_and_align(p)
        if img is None:
            skipped += 1
            t.set_postfix(used=used, skipped=skipped, file=os.path.basename(p))
            continue
        try:
            nh = compute_hash_bits(img).astype(np.uint8).reshape(-1)   # (96,)
            hv = encode_hv(img).astype(np.uint8).reshape(-1)           # (10000,)
            if nh.size != 96 or hv.size != 10000:
                raise ValueError(f"Unexpected vector sizes: NH={nh.size}, HV={hv.size}")
            nh_rows.append(nh)
            hv_rows.append(hv)
            used += 1
            t.set_postfix(used=used, skipped=skipped, file=os.path.basename(p))
        except Exception as e:
            skipped += 1
            t.set_postfix(used=used, skipped=skipped, file=os.path.basename(p))
            t.write(f"[ERROR] {p}: {e}")
    t.close()

    if not nh_rows or not hv_rows:
        print("[ERROR] no valid samples; aborting")
        return

    # Stack NH samples (N,96)
    nh_rows = np.stack(nh_rows, axis=0).astype(np.uint8, copy=False)

    # Build HDIC prototypes and coerce to (K,10000) uint8 {0,1}
    cluster_protos = build_cluster_prototypes(hv_rows, num_clusters=args.hdic_clusters)
    protos = _coerce_protos_to_matrix(cluster_protos, bits=10000)  # (K,10000)

    # Append to packed stores
    nh_store = PackedStore(DB_DIR / "nh_packed", bits=96)
    hd_store = PackedStore(DB_DIR / "hdic_packed", bits=10000)

    print("-> writing NH packed")
    nh_store.append_person(args.id, args.name, nh_rows)
    print("-> writing HDIC packed")
    hd_store.append_person(args.id, args.name, protos)

    print(f"[DONE] enrolled {args.name} ({args.id}); used={used}, skipped={skipped}")

if __name__ == "__main__":
    main()
