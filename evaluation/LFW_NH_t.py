#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuralHash threshold tuning over LFW pairs.

Usage:
  # 1) Precompute NH hashes into a compressed .npz
  python nh_threshold_tuner.py precompute \
      --lfw_dir "D:/.../lfw-deepfunneled/lfw-deepfunneled" \
      --pairs   "D:/.../pairs_new.csv" \
      --out_npz "./nh_hashes_lfw.npz"

  # 2) Load hashes and sweep thresholds 15..50 (inclusive)
  python nh_threshold_tuner.py tune \
      --pairs   "D:/.../pairs_new.csv" \
      --npz     "./nh_hashes_lfw.npz" \
      --tmin 15 --tmax 50
"""

import argparse, csv, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- your real modules ---
from preprocess.align import align_from_array
from neuralhash.adapter import compute_hash_bits

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_image_path(lfw_root: Path, person: str, img_num: str) -> str:
    """
    LFW path pattern: {person}/{person}_{img_num:04d}.jpg
    If img_num can't be cast to int, assume it's a direct path.
    """
    try:
        n = int(img_num)
    except ValueError:
        return img_num
    return str(lfw_root / person / f"{person}_{n:04d}.jpg")

def read_pairs(pairs_csv: Path, lfw_root: Path) -> List[Tuple[str, str, int]]:
    """
    pairs_new.csv rows:
      - 3 cols: person, img1, img2 -> positive (label=1)
      - 4 cols: person1, img1, person2, img2 -> negative (label=0)
    Returns list of (path1, path2, label).
    """
    pairs = []
    with pairs_csv.open("r", newline="") as f:
        rd = csv.reader(f)
        first = next(rd, None)
        if first is None:
            return pairs
        header_like = any(
            h.lower().strip() in {"name", "person", "person1", "person2", "img1", "img2"}
            for h in first
        )
        rows = list(rd) if header_like else [first] + list(rd)

    for row in rows:
        row = [c.strip() for c in row if c is not None]
        if not row: continue
        if len(row) == 3 and all(row):
            person, img1, img2 = row
            p1 = get_image_path(lfw_root, person, img1)
            p2 = get_image_path(lfw_root, person, img2)
            pairs.append((p1, p2, 1))
        elif len(row) == 4 and all(row):
            person1, img1, person2, img2 = row
            p1 = get_image_path(lfw_root, person1, img1)
            p2 = get_image_path(lfw_root, person2, img2)
            pairs.append((p1, p2, 0))
    return pairs

def align_and_hash(img_bgr) -> np.ndarray:
    """
    Align -> 96-bit NeuralHash (uint8, values 0/1), shape (96,)
    """
    face_rgb = align_from_array(img_bgr, output_size=(160,160), normalize=False)
    if face_rgb is None:
        raise ValueError("No face after alignment")
    bits = compute_hash_bits(face_rgb).astype(np.uint8).reshape(-1)
    if bits.size != 96:
        raise ValueError(f"Unexpected NH length: {bits.size}")
    return bits

def hamming96(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

# ------------------------------------------------------------------
# Stage 1: Precompute hashes
# ------------------------------------------------------------------

def cmd_precompute(lfw_dir: Path, pairs_csv: Path, out_npz: Path):
    pairs = read_pairs(pairs_csv, lfw_dir)
    if not pairs:
        print("No pairs found; nothing to precompute.")
        return

    # Collect unique image paths
    uniq = sorted({p for r in pairs for p in (r[0], r[1])})
    print(f"Unique images to hash: {len(uniq)}")

    # Compute hashes
    hashes = []
    ok_paths = []
    skipped = []
    for p in tqdm(uniq, desc="Hashing images"):
        img = cv2.imread(p)
        if img is None:
            skipped.append(p); continue
        try:
            bits = align_and_hash(img)  # (96,)
        except Exception as e:
            skipped.append(p); continue
        ok_paths.append(p)
        hashes.append(bits)

    if not hashes:
        print("No images hashed successfully. Abort.")
        return

    H = np.stack(hashes, axis=0).astype(np.uint8)  # (N,96)
    paths_arr = np.array(ok_paths, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, paths=paths_arr, hashes=H)

    print(f"Saved: {out_npz}  (paths: {len(paths_arr)}, hashes: {H.shape})")
    if skipped:
        print(f"Skipped {len(skipped)} files (not found/align fail). Example: {skipped[:3]}")

# ------------------------------------------------------------------
# Stage 2: Load hashes and tune threshold
# ------------------------------------------------------------------

def load_hash_index(npz_path: Path) -> Tuple[Dict[str, int], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    paths = list(map(str, data["paths"].tolist()))
    hashes = data["hashes"].astype(np.uint8)   # (N,96)
    index = {p: i for i, p in enumerate(paths)}
    return index, hashes

def cmd_tune(pairs_csv: Path, npz_path: Path, tmin: int, tmax: int, step: int):
    index, hashes = load_hash_index(npz_path)

    pairs = read_pairs(pairs_csv, Path("."))  # absolute paths already stored in CSV resolution
    if not pairs:
        print("No pairs to evaluate.")
        return

    # Collect distances for pairs that exist in the index
    dists, labels, missing = [], [], 0
    for p1, p2, y in pairs:
        i1 = index.get(p1); i2 = index.get(p2)
        if i1 is None or i2 is None:
            missing += 1; continue
        d = hamming96(hashes[i1], hashes[i2])
        dists.append(d); labels.append(y)

    if not dists:
        print("No overlapping pairs found in the hash index.")
        return

    dists = np.array(dists, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    print(f"Pairs usable: {len(dists)} (missing: {missing})")

    # Sweep thresholds
    thrs = list(range(tmin, tmax + 1, step))
    print("\nTHresh  Acc     Prec    Rec     F1      FAR     FRR     TP   TN   FP   FN")
    print("----------------------------------------------------------------------------")
    best_row = None
    best_acc = -1.0

    for thr in thrs:
        preds = (dists <= thr).astype(int)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        far = fp / (fp + tn + 1e-12)
        frr = fn / (fn + tp + 1e-12)
        print(f"{thr:5d}  {acc:0.4f}  {prec:0.4f}  {rec:0.4f}  {f1:0.4f}  {far:0.4f}  {frr:0.4f}  {tp:4d} {tn:4d} {fp:4d} {fn:4d}")

        if acc > best_acc:
            best_acc = acc
            best_row = dict(thr=thr, acc=acc, prec=prec, rec=rec, f1=f1, far=far, frr=frr, tp=tp, tn=tn, fp=fp, fn=fn)

    print("\nBest threshold by Accuracy:")
    print(f"  thr={best_row['thr']} | Acc={best_row['acc']:.4f} | Prec={best_row['prec']:.4f} | Rec={best_row['rec']:.4f} | F1={best_row['f1']:.4f}")
    print(f"  FAR={best_row['far']:.4f} | FRR={best_row['frr']:.4f} | TP={best_row['tp']} TN={best_row['tn']} FP={best_row['fp']} FN={best_row['fn']}")
    # If you also want EER search within [tmin,tmax], you can add it here.

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="NeuralHash threshold tuning (LFW).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_p = sub.add_parser("precompute", help="Align and compute NH for all images referenced by pairs; save .npz.")
    ap_p.add_argument("--lfw_dir", type=Path, required=True)
    ap_p.add_argument("--pairs",   type=Path, required=True)
    ap_p.add_argument("--out_npz", type=Path, required=True)

    ap_t = sub.add_parser("tune", help="Load .npz and sweep thresholds.")
    ap_t.add_argument("--pairs", type=Path, required=True)
    ap_t.add_argument("--npz",   type=Path, required=True)
    ap_t.add_argument("--tmin",  type=int, default=15)
    ap_t.add_argument("--tmax",  type=int, default=50)
    ap_t.add_argument("--step",  type=int, default=1)

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "precompute":
        cmd_precompute(args.lfw_dir, args.pairs, args.out_npz)
    elif args.cmd == "tune":
        cmd_tune(args.pairs, args.npz, args.tmin, args.tmax, args.step)

if __name__ == "__main__":
    main()
