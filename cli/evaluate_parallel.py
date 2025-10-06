import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json, csv
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
    probe_hv   = encode_hv(probe)

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
    correct_accepts = 0

    # Watchlist per-image progress (position=3 to allow nested outer sweeps)
    wl_pbar = tqdm(watchlist_imgs,
                   desc=f"watchlist w_nh={w_nh} Tnh={Tnh} Thdic={Thdic}",
                   unit="img", position=3, leave=False, total=total_watchlist)
    processed_wl = 0
    for img in wl_pbar:
        gt = get_person_id_from_path(img)
        decision, pred = evaluate_probe(img, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both)
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

    # Background per-image progress
    processed_bg = 0
    if total_bg > 0:
        bg_pbar = tqdm(background_imgs,
                       desc=f"background w_nh={w_nh} Tnh={Tnh} Thdic={Thdic}",
                       unit="img", position=3, leave=False, total=total_bg)
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

    return {"FRR": FRR, "FMR": FMR, "FAR": FAR}


def sweep(dataset, nh_db, hdic_db, Tnh_vals, Thdic_vals, weight_vals, fused_th, require_both, background=None):
    results = {}
    # Outer progress: weight sweep
    for w_nh in tqdm(weight_vals, desc="w_nh sweep", position=0):
        w_hdic = 1.0 - w_nh
        # Next level: Tnh sweep
        for Tnh in tqdm(Tnh_vals, desc=f"Tnh sweep (w_nh={w_nh})", position=1, leave=False):
            # Inner: Thdic sweep
            for Thdic in tqdm(Thdic_vals, desc=f"Thdic sweep (Tnh={Tnh})", position=2, leave=False):
                m = compute_metrics(dataset, nh_db, hdic_db, Tnh, Thdic, w_nh, w_hdic, fused_th, require_both, background)
                results[(Tnh, Thdic, w_nh)] = m
                tqdm.write(f"[w_nh={w_nh:.2f} Tnh={Tnh} Thdic={Thdic}] FRR={m['FRR']:.4f}, FMR={m['FMR']:.4f}, FAR={m['FAR']:.4f}")
    return results


def compute_eer(results, Tnh_vals, Thdic_vals, weight_vals, out_dir):
    eer_records = []
    for w_nh in weight_vals:
        for Tnh in Tnh_vals:
            curve = [(results[(Tnh, Thdic, w_nh)]["FAR"], results[(Tnh, Thdic, w_nh)]["FRR"], Thdic)
                     for Thdic in Thdic_vals]
            diffs = [abs(far - frr) for far, frr, _ in curve]
            idx = int(np.argmin(diffs))
            far, frr, thdic = curve[idx]
            eer = (far + frr) / 2
            eer_records.append({"w_nh": w_nh, "Tnh": Tnh, "Thdic@EER": thdic, "FAR": far, "FRR": frr, "EER": eer})

    path = os.path.join(out_dir, "parallel_eer.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eer_records[0].keys())
        writer.writeheader(); writer.writerows(eer_records)
    return eer_records


def plot_heatmaps(results, Tnh_vals, Thdic_vals, weight_vals, out_dir):
    for w_nh in weight_vals:
        acc = np.zeros((len(Tnh_vals), len(Thdic_vals)))
        for i, Tnh in enumerate(Tnh_vals):
            for j, Thdic in enumerate(Thdic_vals):
                m = results[(Tnh, Thdic, w_nh)]
                acc[i, j] = 1 - (m["FAR"] + m["FRR"]) / 2

        plt.figure()
        plt.imshow(acc, origin="lower", aspect="auto",
                   extent=[min(Thdic_vals), max(Thdic_vals),
                           min(Tnh_vals), max(Tnh_vals)])
        plt.colorbar(label="Accuracy proxy")
        plt.xlabel("Thdic"); plt.ylabel("Tnh")
        plt.title(f"Accuracy Heatmap (w_nh={w_nh}, w_hdic={1-w_nh})")
        plt.savefig(os.path.join(out_dir, f"heatmap_w{w_nh}.png"), dpi=200)
        plt.close()


def plot_roc_det(results, Tnh_vals, Thdic_vals, weight_vals, out_dir, eer_records):
    for w_nh in weight_vals:
        # ROC
        plt.figure()
        for Tnh in Tnh_vals:
            curve = sorted([(results[(Tnh, Thdic, w_nh)]["FAR"], 1 - results[(Tnh, Thdic, w_nh)]["FRR"])
                            for Thdic in Thdic_vals])
            xs, ys = zip(*curve)
            plt.plot(xs, ys, marker="o", label=f"Tnh={Tnh}")
            rec = next(r for r in eer_records if r["Tnh"] == Tnh and abs(r["w_nh"]-w_nh)<1e-6)
            plt.plot(rec["FAR"], 1 - rec["FRR"], "r*", markersize=12)
        plt.xlabel("FAR"); plt.ylabel("TPR (1-FRR)")
        plt.title(f"ROC curves (w_nh={w_nh})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f"roc_w{w_nh}.png"), dpi=200)
        plt.close()

        # DET
        plt.figure()
        for Tnh in Tnh_vals:
            curve = sorted([(results[(Tnh, Thdic, w_nh)]["FAR"], results[(Tnh, Thdic, w_nh)]["FRR"])
                            for Thdic in Thdic_vals])
            xs, ys = zip(*curve)
            plt.plot(xs, ys, marker="o", label=f"Tnh={Tnh}")
            rec = next(r for r in eer_records if r["Tnh"] == Tnh and abs(r["w_nh"]-w_nh)<1e-6)
            plt.plot(rec["FAR"], rec["FRR"], "r*", markersize=12)
        plt.xlabel("FAR"); plt.ylabel("FRR")
        plt.title(f"DET curves (w_nh={w_nh})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f"det_w{w_nh}.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/test")
    parser.add_argument("--background", default=None)
    parser.add_argument("--out", default="plots/parallel")
    parser.add_argument("--tnh", type=int, nargs=3, default=[10, 50, 10])
    parser.add_argument("--thdic", type=int, nargs=3, default=[2000, 4000, 200])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--fused_th", type=float, default=0.7)
    parser.add_argument("--require_both", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    Tnh_vals = list(range(args.tnh[0], args.tnh[1] + 1, args.tnh[2]))
    Thdic_vals = list(range(args.thdic[0], args.thdic[1] + 1, args.thdic[2]))

    nh_db, hdic_db = load_nh_db(), load_hdic_db()
    results = sweep(args.dataset, nh_db, hdic_db, Tnh_vals, Thdic_vals,
                    args.weights, args.fused_th, args.require_both, args.background)

    # Best combination
    best = {"BER": 1e9}
    for (Tnh, Thdic, w_nh), m in results.items():
        BER = (m["FAR"] + m["FRR"]) / 2
        if BER < best["BER"]:
            best = {"Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": 1-w_nh, "metrics": m, "BER": BER}

    eer_records = compute_eer(results, Tnh_vals, Thdic_vals, args.weights, args.out)
    best["EERs"] = eer_records

    with open(os.path.join(args.out, "best_thresholds.json"), "w") as f:
        json.dump(best, f, indent=2)

    plot_heatmaps(results, Tnh_vals, Thdic_vals, args.weights, args.out)
    plot_roc_det(results, Tnh_vals, Thdic_vals, args.weights, args.out, eer_records)

    print("[DONE] Parallel evaluation complete.")
    print(f"Best: Tnh={best['Tnh']}, Thdic={best['Thdic']}, w_nh={best['w_nh']}, w_hdic={best['w_hdic']}, BER={best['BER']:.4f}")
    print(f"Results saved in: {args.out}")


if __name__ == "__main__":
    main()
