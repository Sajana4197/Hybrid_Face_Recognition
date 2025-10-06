import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from hdic.adapter import encode_hv
from fusion.parallel import best_neuralhash_distance, best_hdic_distance, decide_parallel


def get_person_id_from_path(path):
    return os.path.basename(os.path.dirname(path))


def iter_images(root):
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                yield os.path.join(dp, f)


def evaluate_probe(img_path, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both):
    probe = load_and_align(img_path)
    if probe is None:
        return "skip", None

    probe_bits = compute_hash_bits(probe)
    probe_hv = encode_hv(probe)

    nh_d = best_neuralhash_distance(probe_bits, nh_db)
    hd_d = best_hdic_distance(probe_hv, hdic_db)

    best_pid, _ = decide_parallel(
        nh_d, hd_d,
        Tnh_reject=Tnh,
        Thdic_accept=Thdic,
        w_nh=w_nh,
        w_hdic=w_hdic,
        fused_threshold=fused_th,
        require_both_modalities=require_both,
    )

    if best_pid is None:
        return "reject", None
    return "accept", best_pid


def compute_metrics(dataset, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both, background=None):
    watchlist_imgs = list(iter_images(dataset))
    background_imgs = list(iter_images(background)) if background else []

    total_watchlist = len(watchlist_imgs)
    total_bg = len(background_imgs)
    false_rejects, false_accepts_bg, false_matches = 0, 0, 0

    # Watchlist per-image progress
    wl_pbar = tqdm(watchlist_imgs,
                   desc=f"watchlist w_nh={w_nh:.2f} Tnh={Tnh} Thdic={Thdic}",
                   unit="img", position=2, leave=False, total=total_watchlist)
    processed_wl = 0
    for img in wl_pbar:
        gt = get_person_id_from_path(img)
        decision, pred = evaluate_probe(img, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both)
        processed_wl += 1
        if decision == "skip":
            total_watchlist -= 1
            wl_pbar.set_postfix(skipped=processed_wl - (false_rejects + false_matches))
            continue
        if decision == "reject":
            false_rejects += 1
        elif pred != gt:
            false_matches += 1
        wl_pbar.set_postfix(fr=false_rejects, fm=false_matches)

    # Background per-image progress
    if total_bg > 0:
        bg_pbar = tqdm(background_imgs,
                       desc=f"background w_nh={w_nh:.2f} Tnh={Tnh} Thdic={Thdic}",
                       unit="img", position=2, leave=False, total=total_bg)
        processed_bg = 0
        for img in bg_pbar:
            decision, pred = evaluate_probe(img, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both)
            processed_bg += 1
            if decision == "skip":
                total_bg -= 1
                bg_pbar.set_postfix(skipped=processed_bg - (false_accepts_bg))
                continue
            if decision == "accept":
                false_accepts_bg += 1
            bg_pbar.set_postfix(fa=false_accepts_bg)

    FRR = false_rejects / total_watchlist if total_watchlist > 0 else 0
    FMR = false_matches / total_watchlist if total_watchlist > 0 else 0
    FAR = false_accepts_bg / total_bg if total_bg > 0 else FMR
    return FAR, FRR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/test")
    parser.add_argument("--background", default=None)
    parser.add_argument("--out", default="plots/parallel_weight_sweep")
    parser.add_argument("--tnh", type=int, default=30)
    parser.add_argument("--thdic", type=int, default=3100)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--fused_th", type=float, default=0.7)
    parser.add_argument("--require_both", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nh_db, hdic_db = load_nh_db(), load_hdic_db()

    FARs, FRRs = [], []

    print(f"[INFO] Evaluating for Tnh={args.tnh}, Thdic={args.thdic}")
    for w_nh in tqdm(args.weights, desc="Weight sweep"):
        w_hdic = 1 - w_nh
        FAR, FRR = compute_metrics(args.dataset, nh_db, hdic_db, args.tnh, args.thdic, w_nh, w_hdic,
                                   args.fused_th, args.require_both, args.background)
        FARs.append(FAR)
        FRRs.append(FRR)
        print(f"  w_nh={w_nh:.2f}, w_hdic={w_hdic:.2f} → FAR={FAR:.4f}, FRR={FRR:.4f}")

    # Plot
    plt.figure()
    plt.plot(args.weights, FARs, marker="o", label="FAR")
    plt.plot(args.weights, FRRs, marker="o", label="FRR")
    plt.xlabel("w_nh (NeuralHash weight)")
    plt.ylabel("Rate")
    plt.title(f"FAR & FRR vs w_nh (Tnh={args.tnh}, Thdic={args.thdic})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.out, "far_frr_vs_weight.png"), dpi=200)
    plt.close()

    # Save table
    data = [{"w_nh": w, "w_hdic": 1-w, "FAR": f, "FRR": r} for w, f, r in zip(args.weights, FARs, FRRs)]
    with open(os.path.join(args.out, "far_frr_vs_weight.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"[DONE] Plot saved to {os.path.join(args.out, 'far_frr_vs_weight.png')}")


if __name__ == "__main__":
    main()
