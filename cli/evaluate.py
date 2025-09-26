import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from fusion.cascade import shortlist_by_neuralhash, hdic_confirm
from hdic.adapter import encode_hv

def run_cascade(img_path, target_id, label, nh_db, hdic_db, Tnh, Thdic, K=5):
    probe = load_and_align(img_path)
    if probe is None:
        return None  # skip if no face

    nh_bits = compute_hash_bits(probe)
    shortlist, rejected = shortlist_by_neuralhash(nh_bits, nh_db, Tnh, K)

    if rejected:
        decision = "REJECT"
    else:
        probe_hv = encode_hv(probe)
        best_dist, best_pid = 1e9, None
        for pid, _ in shortlist:
            d, _ = hdic_confirm(probe_hv, pid, hdic_db)
            if d < best_dist:
                best_dist, best_pid = d, pid
        decision = "MATCH" if best_pid == target_id and best_dist < Thdic else "REJECT"

    if decision == "MATCH" and label == "genuine":
        return "TP"
    elif decision == "MATCH" and label == "impostor":
        return "FP"
    elif decision == "REJECT" and label == "genuine":
        return "FN"
    elif decision == "REJECT" and label == "impostor":
        return "TN"
    return None

def compute_metrics(results):
    TP = results.count("TP")
    FP = results.count("FP")
    TN = results.count("TN")
    FN = results.count("FN")

    tar = TP / (TP + FN + 1e-9)
    far = FP / (FP + TN + 1e-9)
    frr = FN / (TP + FN + 1e-9)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "TAR": tar, "FAR": far, "FRR": frr, "Accuracy": acc}

def sweep_thresholds(pairs, nh_db, hdic_db, Tnh=30, K=5, th_start=1000, th_end=4000, th_step=500):
    thresholds, fars, frrs, accs = [], [], [], []
    for Thdic in range(th_start, th_end + 1, th_step):
        results = []
        # show per-threshold progress over pairs
        for row in tqdm(pairs, desc=f"Eval Thdic={Thdic}", unit="pair", leave=False):
            r = run_cascade(row["image_path"], row["target_person_id"], row["label"],
                            nh_db, hdic_db, Tnh, Thdic, K)
            if r: results.append(r)
        m = compute_metrics(results)
        thresholds.append(Thdic)
        fars.append(m["FAR"])
        frrs.append(m["FRR"])
        accs.append(m["Accuracy"])
        print(f"Thdic={Thdic}, Acc={m['Accuracy']:.4f}, TAR={m['TAR']:.3f}, FAR={m['FAR']:.3f}, FRR={m['FRR']:.3f}")
    return thresholds, fars, frrs, accs

def plot_curves(thresholds, fars, frrs, accs, out_dir="plots"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # FAR & FRR vs Threshold
    plt.figure()
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.xlabel("HDIC Threshold")
    plt.ylabel("Rate")
    plt.title("FAR & FRR vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "far_frr_vs_threshold.png"))

    # Accuracy vs Threshold
    plt.figure()
    plt.plot(thresholds, accs, label="Accuracy", color="green")
    plt.xlabel("HDIC Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_vs_threshold.png"))

    # DET curve (FAR vs FRR)
    plt.figure()
    plt.plot(fars, frrs, marker="o")
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET Curve (FAR vs FRR)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "det_curve.png"))

    print(f"[SAVED] Plots saved to {out_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="CSV file with [image,label,target_person_id]")
    parser.add_argument("--Tnh", type=int, default=30, help="NeuralHash reject threshold")
    parser.add_argument("--K", type=int, default=5, help="NeuralHash shortlist size")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    args = parser.parse_args()

    nh_db = load_nh_db()
    hdic_db = load_hdic_db()

    with open(args.pairs, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = list(reader)

    if args.sweep:
        thresholds, fars, frrs, accs = sweep_thresholds(pairs, nh_db, hdic_db, args.Tnh, args.K)
        plot_curves(thresholds, fars, frrs, accs)
    else:
        results = []
        # show progress for evaluation of pairs
        for row in tqdm(pairs, desc="Evaluating pairs", unit="pair"):
            r = run_cascade(row["image_path"], row["target_person_id"], row["label"],
                            nh_db, hdic_db, args.Tnh, 5000, args.K)
            if r: results.append(r)
        m = compute_metrics(results)
        print(f"Results @ Thdic=5000, Tnh={args.Tnh}")
        print(m)

if __name__ == "__main__":
    main()
