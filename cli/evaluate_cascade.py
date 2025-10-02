import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure imports work

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from hdic.adapter import encode_hv
from fusion.cascade import shortlist_by_neuralhash, hdic_confirm


def get_person_id_from_path(path):
    return os.path.basename(os.path.dirname(path))


def iter_images(root):
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                yield os.path.join(dp, f)


def evaluate_probe(img_path, nh_db, hdic_db, Tnh, Thdic, K=5):
    probe = load_and_align(img_path)
    if probe is None:
        return "skip", None

    # Step 1: NeuralHash shortlist
    probe_bits = compute_hash_bits(probe)
    shortlist, nh_rejected = shortlist_by_neuralhash(probe_bits, nh_db, Tnh, K)
    if nh_rejected or len(shortlist) == 0:
        return "reject", None

    # Step 2: HDIC confirm
    probe_hv = encode_hv(probe)
    best_person, best_dist = None, 1e9
    for pid, _ in shortlist:
        dist, _ = hdic_confirm(probe_hv, pid, hdic_db)
        if dist < best_dist:
            best_dist, best_person = dist, pid

    if best_person is not None and best_dist < Thdic:
        return "accept", best_person
    else:
        return "reject", None


def compute_metrics(dataset, nh_db, hdic_db, Tnh, Thdic, K=5, background=None):
    watchlist_imgs = list(iter_images(dataset))
    background_imgs = list(iter_images(background)) if background else []

    total_watchlist = len(watchlist_imgs)
    total_bg = len(background_imgs)

    false_rejects, false_accepts_bg, false_matches = 0, 0, 0
    correct_accepts = 0

    # watchlist progress
    wl_pbar = tqdm(watchlist_imgs, desc=f"watchlist Tnh={Tnh} Thdic={Thdic}", unit="img", position=2, leave=False, total=total_watchlist)
    processed_wl = 0
    for img in wl_pbar:
        gt = get_person_id_from_path(img)
        decision, pred = evaluate_probe(img, nh_db, hdic_db, Tnh, Thdic, K)
        processed_wl += 1
        if decision == "skip":
            total_watchlist -= 1
            wl_pbar.set_postfix(skipped=processed_wl - (false_rejects + correct_accepts + false_matches))
            continue
        if decision == "reject":
            false_rejects += 1
        elif pred == gt:
            correct_accepts += 1
        else:
            false_matches += 1
        wl_pbar.set_postfix(fr=false_rejects, ca=correct_accepts, fm=false_matches)

    # background progress
    processed_bg = 0
    if total_bg > 0:
        bg_pbar = tqdm(background_imgs, desc=f"background Tnh={Tnh} Thdic={Thdic}", unit="img", position=2, leave=False, total=total_bg)
        for img in bg_pbar:
            decision, pred = evaluate_probe(img, nh_db, hdic_db, Tnh, Thdic, K)
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
    FAR = false_accepts_bg / total_bg if total_bg > 0 else FMR  # fallback

    return {"FRR": FRR, "FMR": FMR, "FAR": FAR}


def sweep(dataset, nh_db, hdic_db, Tnh_vals, Thdic_vals, K=5, background=None):
    results = {}
    # outer progress: sweep over Tnh
    for Tnh in tqdm(Tnh_vals, desc="Tnh sweep", position=0):
        # inner progress: sweep over Thdic for this Tnh
        for Thdic in tqdm(Thdic_vals, desc=f"Tnh={Tnh} Thdic sweep", position=1, leave=False):
            m = compute_metrics(dataset, nh_db, hdic_db, Tnh, Thdic, K, background)
            results[(Tnh, Thdic)] = m
            tqdm.write(f"[Tnh={Tnh}, Thdic={Thdic}] FRR={m['FRR']:.4f}, FMR={m['FMR']:.4f}, FAR={m['FAR']:.4f}")
    return results


def compute_eer(results, Tnh_vals, Thdic_vals, out_dir):
    eer_records = []
    for Tnh in Tnh_vals:
        curve = [(results[(Tnh, Thdic)]["FAR"], results[(Tnh, Thdic)]["FRR"], Thdic) 
                 for Thdic in Thdic_vals]
        # Find point where FAR ≈ FRR
        diffs = [abs(far - frr) for far, frr, _ in curve]
        idx = int(np.argmin(diffs))
        far, frr, thdic = curve[idx]
        eer = (far + frr) / 2
        eer_records.append({"Tnh": Tnh, "Thdic@EER": thdic, "FAR": far, "FRR": frr, "EER": eer})

    # Save CSV
    import csv
    path = os.path.join(out_dir, "cascade_eer.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eer_records[0].keys())
        writer.writeheader()
        writer.writerows(eer_records)

    return eer_records


def plot_roc_det(results, Tnh_vals, Thdic_vals, out_dir, eer_records):
    # ROC
    plt.figure()
    for Tnh in Tnh_vals:
        curve = sorted([(results[(Tnh, Thdic)]["FAR"], 1 - results[(Tnh, Thdic)]["FRR"]) 
                        for Thdic in Thdic_vals])
        xs, ys = zip(*curve)
        plt.plot(xs, ys, marker="o", label=f"Tnh={Tnh}")
        # EER marker
        rec = next(r for r in eer_records if r["Tnh"] == Tnh)
        plt.plot(rec["FAR"], 1 - rec["FRR"], "r*", markersize=12)

    plt.xlabel("FAR")
    plt.ylabel("TPR (1 - FRR)")
    plt.title("ROC curves (cascade)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "cascade_roc.png"), dpi=200)
    plt.close()

    # DET (FAR vs FRR)
    plt.figure()
    for Tnh in Tnh_vals:
        curve = sorted([(results[(Tnh, Thdic)]["FAR"], results[(Tnh, Thdic)]["FRR"]) 
                        for Thdic in Thdic_vals])
        xs, ys = zip(*curve)
        plt.plot(xs, ys, marker="o", label=f"Tnh={Tnh}")
        rec = next(r for r in eer_records if r["Tnh"] == Tnh)
        plt.plot(rec["FAR"], rec["FRR"], "r*", markersize=12)

    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET curves (cascade)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "cascade_det.png"), dpi=200)
    plt.close()


def plot_best_neighborhood(results, best_Tnh, best_Thdic, Tnh_vals, Thdic_vals, out_dir):
    # Sweep Thdic at fixed best Tnh
    xs, FARs, FRRs = [], [], []
    for Thdic in Thdic_vals:
        m = results[(best_Tnh, Thdic)]
        xs.append(Thdic)
        FARs.append(m["FAR"])
        FRRs.append(m["FRR"])
    plt.figure()
    plt.plot(xs, FARs, label="FAR")
    plt.plot(xs, FRRs, label="FRR")
    plt.xlabel("Thdic")
    plt.ylabel("Rate")
    plt.title(f"FAR/FRR vs Thdic (Tnh={best_Tnh})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "best_far_frr_vs_thdic.png"), dpi=200)
    plt.close()

    # Sweep Tnh at fixed best Thdic
    xs, FARs, FRRs = [], [], []
    for Tnh in Tnh_vals:
        m = results[(Tnh, best_Thdic)]
        xs.append(Tnh)
        FARs.append(m["FAR"])
        FRRs.append(m["FRR"])
    plt.figure()
    plt.plot(xs, FARs, label="FAR")
    plt.plot(xs, FRRs, label="FRR")
    plt.xlabel("Tnh")
    plt.ylabel("Rate")
    plt.title(f"FAR/FRR vs Tnh (Thdic={best_Thdic})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "best_far_frr_vs_tnh.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/test")
    parser.add_argument("--background", default=None)
    parser.add_argument("--out", default="plots/cascade_new")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tnh", type=int, nargs=3, default=[10, 50, 10])
    parser.add_argument("--thdic", type=int, nargs=3, default=[2000, 4000, 200])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    Tnh_vals = list(range(args.tnh[0], args.tnh[1] + 1, args.tnh[2]))
    Thdic_vals = list(range(args.thdic[0], args.thdic[1] + 1, args.thdic[2]))

    nh_db, hdic_db = load_nh_db(), load_hdic_db()
    results = sweep(args.dataset, nh_db, hdic_db, Tnh_vals, Thdic_vals, args.k, args.background)

    # Find best thresholds (min BER)
    best = {"BER": 1e9}
    for (Tnh, Thdic), m in results.items():
        BER = (m["FAR"] + m["FRR"]) / 2
        if BER < best["BER"]:
            best = {"Tnh": Tnh, "Thdic": Thdic, "metrics": m, "BER": BER}

    # Compute EERs
    eer_records = compute_eer(results, Tnh_vals, Thdic_vals, args.out)

    # Save best thresholds
    best["EERs"] = eer_records
    with open(os.path.join(args.out, "best_thresholds.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Plots
    plot_roc_det(results, Tnh_vals, Thdic_vals, args.out, eer_records)
    plot_best_neighborhood(results, best["Tnh"], best["Thdic"], Tnh_vals, Thdic_vals, args.out)

    print("[DONE] Evaluation complete.")
    print(f"Best thresholds: Tnh={best['Tnh']}, Thdic={best['Thdic']}, BER={best['BER']:.4f}")
    print(f"Results saved in: {args.out}")


if __name__ == "__main__":
    main()
