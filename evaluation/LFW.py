#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion-only LFW evaluation with EER sweeping over weights, plus Threshold-EER,
ROC, and DET curves for the best-EER weight.

Requires the same modules as your fusion logic:
  - preprocess.align.align_from_array
  - neuralhash.adapter.compute_hash_bits
  - hdic.feature_extractor.generate_embedding2
  - hdic.encode_hv.encode_embedding_to_hv
"""

import os, csv, random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# ==== your real modules ====
from preprocess.align import align_from_array
from neuralhash.adapter import compute_hash_bits
from hdic.feature_extractor import generate_embedding2
from hdic.encode_hv import encode_embedding_to_hv

# =========================
# CONFIG
# =========================
LFW_DIR    = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_FILE = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\pairs_new.csv"

WEIGHT_GRID = [i/10.0 for i in range(0, 11)]  # 0.00..1.00 step 0.10
RANDOM_SEED = 1337

# =========================
# UTILITIES
# =========================

def get_image_path(lfw_root: str, person: str, img_num: str) -> str:
    try:
        n = int(img_num)
        name = f"{person}_{n:04d}.jpg"
        return str(Path(lfw_root) / person / name)
    except ValueError:
        return img_num

def read_bgr(path: str):
    return cv2.imread(path)

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def compute_feats(img_bgr):
    face_rgb = align_from_array(img_bgr, output_size=(160,160), normalize=False)
    if face_rgb is None:
        raise ValueError("No face after alignment")
    bits96 = compute_hash_bits(face_rgb).astype(np.uint8).reshape(-1)
    emb512 = generate_embedding2(face_rgb)
    hv10k  = encode_embedding_to_hv(emb512).astype(np.uint8).reshape(-1)
    if bits96.size != 96 or hv10k.size != 10000:
        raise ValueError("Unexpected feature size")
    return bits96, hv10k

def safe_feats(path, cache: dict):
    if path in cache: return cache[path]
    img = read_bgr(path)
    if img is None:
        cache[path] = None
        return None
    try:
        val = compute_feats(img)
    except Exception:
        val = None
    cache[path] = val
    return val

def read_pairs(pairs_csv: str, lfw_root: str):
    pairs = []
    with open(pairs_csv, "r", newline="") as f:
        rd = csv.reader(f)
        first = next(rd, None)
        if first is None: return pairs
        header_like = any(h.lower().strip() in {"name","person","person1","person2","img1","img2"} for h in first)
        rows = list(rd) if header_like else [first] + list(rd)
    for row in rows:
        row = [c.strip() for c in row if c is not None]
        if not row: continue
        if len(row) == 3 and all(row):
            person, img1, img2 = row
            p1 = get_image_path(lfw_root, person, img1)
            p2 = get_image_path(lfw_root, person, img2)
            pairs.append((p1,p2,1))
        elif len(row) == 4 and all(row):
            p1 = get_image_path(lfw_root, row[0], row[1])
            p2 = get_image_path(lfw_root, row[2], row[3])
            pairs.append((p1,p2,0))
    return pairs

# Inverse normal CDF (probit) without SciPy — Acklam’s approximation
def probit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1-1e-12)
    # coefficients:
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    x = np.empty_like(p, dtype=np.float64)
    # lower region
    mask = p < plow
    q = np.sqrt(-2*np.log(p[mask]))
    x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    # central region
    mask = (p >= plow) & (p <= phigh)
    q = p[mask] - 0.5
    r = q*q
    x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    # upper region
    mask = p > phigh
    q = np.sqrt(-2*np.log(1 - p[mask]))
    x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return x

# Compute FAR, FRR and confusion for a threshold on *distance* (match if dist <= thr)
def metrics_at_threshold(dist: np.ndarray, labels: np.ndarray, thr: float):
    preds = (dist <= thr).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    far = fp / (fp + tn + 1e-12)
    frr = fn / (fn + tp + 1e-12)
    return acc, prec, rec, f1, far, frr, tp, tn, fp, fn

# Find EER by scanning thresholds and linearly interpolating around the crossing
def find_eer(dist: np.ndarray, labels: np.ndarray, thr_grid: np.ndarray):
    fars, frrs = [], []
    for thr in thr_grid:
        _, _, _, _, far, frr, *_ = metrics_at_threshold(dist, labels, thr)
        fars.append(far); frrs.append(frr)
    fars = np.asarray(fars); frrs = np.asarray(frrs)
    diffs = np.abs(fars - frrs)
    i = int(np.argmin(diffs))
    eer = 0.5*(fars[i] + frrs[i])
    thr_eer = float(thr_grid[i])
    return eer, thr_eer, fars, frrs

# =========================
# MAIN
# =========================

def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    print("Loading pairs ...")
    pairs = read_pairs(PAIRS_FILE, LFW_DIR)
    print(f"Pairs parsed: {len(pairs)}")
    if not pairs:
        print("No pairs."); return

    # Precompute features
    print("Precomputing features (NH 96b + HDIC 10kb) for unique images ...")
    paths = sorted({p for x in pairs for p in (x[0], x[1])})
    cache = {}
    for p in tqdm(paths, desc="Images"):
        _ = safe_feats(p, cache)

    # Build fused-ready arrays
    dnh_list, dhv_list, y_list = [], [], []
    skipped = 0
    for p1, p2, y in pairs:
        f1 = cache.get(p1); f2 = cache.get(p2)
        if f1 is None or f2 is None:
            skipped += 1; continue
        b1, h1 = f1; b2, h2 = f2
        dnh = hamming(b1,b2)          # 0..96
        dhv = hamming(h1,h2)          # 0..10000
        dnh_list.append(dnh/96.0)     # normalize to [0,1]
        dhv_list.append(dhv/10000.0)  # normalize to [0,1]
        y_list.append(y)
    print(f"Usable pairs: {len(y_list)} (skipped {skipped})")
    if len(y_list) == 0:
        print("Nothing usable."); return

    dnh = np.array(dnh_list, dtype=np.float64)
    dhv = np.array(dhv_list, dtype=np.float64)
    y   = np.array(y_list,   dtype=np.int32)

    results = []  # table rows
    best_idx = None
    best_eer = 1.0

    # Sweep weights
    print("\n==================== RESULTS TABLE ====================")
    print("w_nh  w_hdic   EER     Acc@EER   Prec    Rec     F1      FAR     FRR")
    print("---------------------------------------------------------------")
    for w_nh in WEIGHT_GRID:
        w_hdic = 1.0 - w_nh
        fused = w_nh*dnh + w_hdic*dhv  # distance in [0,1]
        thr_grid = np.linspace(fused.min(), fused.max(), 4001)  # fine grid

        eer, thr_eer, fars, frrs = find_eer(fused, y, thr_grid)
        acc, prec, rec, f1, far, frr, tp, tn, fp, fn = metrics_at_threshold(fused, y, thr_eer)

        results.append({
            "w_nh": w_nh, "w_hdic": w_hdic,
            "eer": eer, "thr_eer": thr_eer,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "far": far, "frr": frr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "fars_curve": fars, "frrs_curve": frrs, "fused": fused, "labels": y
        })

        print(f"{w_nh:0.2f}  {w_hdic:0.2f}  {eer:0.4f}  {acc:0.4f}  {prec:0.4f}  {rec:0.4f}  {f1:0.4f}  {far:0.4f}  {frr:0.4f}")

        if eer < best_eer:
            best_eer = eer
            best_idx = len(results)-1

    # Summary like your screenshot
    by_acc = max(results, key=lambda r: r["acc"])
    by_eer = min(results, key=lambda r: r["eer"])
    by_f1  = max(results, key=lambda r: r["f1"])

    print("\n======================== SUMMARY ========================")
    print(f"Best Accuracy at EER: {by_acc['acc']:.4f} at w_nh={by_acc['w_nh']:.2f}, w_hdic={by_acc['w_hdic']:.2f}")
    print(f"Best EER: {by_eer['eer']:.4f} at w_nh={by_eer['w_nh']:.2f}, w_hdic={by_eer['w_hdic']:.2f}")
    print(f"Best F1 Score: {by_f1['f1']:.4f} at w_nh={by_f1['w_nh']:.2f}, w_hdic={by_f1['w_hdic']:.2f}")

    r = results[best_idx]
    print("\nDetailed metrics at best EER (w_nh={:.2f}):".format(r["w_nh"]))
    print(f" - Accuracy : {r['acc']:.4f}")
    print(f" - Precision: {r['prec']:.4f}")
    print(f" - Recall   : {r['rec']:.4f}")
    print(f" - F1 Score : {r['f1']:.4f}")
    print(f" - FAR      : {r['far']:.4f}")
    print(f" - FRR      : {r['frr']:.4f}")
    print(f" - TP: {r['tp']}, TN: {r['tn']}")
    print(f" - FP: {r['fp']}, FN: {r['fn']}")

    # =========================
    # PLOTS for best-EER weight
    # =========================
    out_dir = Path(".")
    w_nh = r["w_nh"]; w_hdic = r["w_hdic"]
    fused = r["fused"]; labels = r["labels"]

    # Recompute full curves for plotting
    thr_grid = np.linspace(fused.min(), fused.max(), 4001)
    fars = []; frrs = []; tprs = []; fprs = []
    for thr in thr_grid:
        _, _, _, _, far, frr, tp, tn, fp, fn = metrics_at_threshold(fused, labels, thr)
        fars.append(far); frrs.append(frr)
        fprs.append(far)
        tprs.append(1 - frr)

    fars = np.array(fars); frrs = np.array(frrs); fprs = np.array(fprs); tprs = np.array(tprs)
    # EER locator
    diffs = np.abs(fars - frrs)
    i_eer = int(np.argmin(diffs))
    thr_eer = float(thr_grid[i_eer]); eer = 0.5*(fars[i_eer] + frrs[i_eer])

    # 1) Threshold vs FAR/FRR (EER point)
    plt.figure()
    plt.plot(thr_grid, fars, label="FAR")
    plt.plot(thr_grid, frrs, label="FRR")
    plt.axvline(thr_eer, linestyle="--")
    plt.axhline(eer, linestyle="--")
    plt.title(f"Threshold vs FAR/FRR (EER={eer:.4f}) | w_nh={w_nh:.2f}, w_hdic={w_hdic:.2f}")
    plt.xlabel("Fusion Distance Threshold")
    plt.ylabel("Rate")
    plt.legend()
    out1 = out_dir / f"fusion_threshold_EER_w{int(w_nh*100)}.png"
    plt.savefig(out1, bbox_inches="tight"); plt.close()

    # 2) ROC (TPR vs FPR)
    plt.figure()
    plt.plot(fprs, tprs)
    plt.scatter([fprs[i_eer]], [tprs[i_eer]])
    plt.title(f"ROC (AUC not computed) | EER@thr={thr_eer:.4f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    out2 = out_dir / f"fusion_ROC_w{int(w_nh*100)}.png"
    plt.savefig(out2, bbox_inches="tight"); plt.close()

    # 3) DET (probit axes)
    plt.figure()
    # convert to standard normal deviate
    x = probit(np.array(fprs))
    ydet = probit(1 - np.array(tprs))  # FNR = 1-TPR
    xe = probit(np.array([fprs[i_eer]])); ye = probit(np.array([1 - tprs[i_eer]]))
    plt.plot(x, ydet)
    plt.scatter(xe, ye)
    plt.title(f"DET Curve | EER={eer:.4f}")
    plt.xlabel("probit(FPR)")
    plt.ylabel("probit(FNR)")
    out3 = out_dir / f"fusion_DET_w{int(w_nh*100)}.png"
    plt.savefig(out3, bbox_inches="tight"); plt.close()

    print("\nSaved plots:")
    print(f" - {out1}")
    print(f" - {out2}")
    print(f" - {out3}")

    # NOTE: If you want to use this EER threshold in your live matcher:
    #   Sfinal = 1 - fused_distance
    #   fused_th_for_service = 1 - thr_eer
    # and keep your Tnh/Thdic per-person logic unchanged.

if __name__ == "__main__":
    main()
