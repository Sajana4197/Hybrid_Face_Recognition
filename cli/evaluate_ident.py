import argparse, os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from fusion.cascade import shortlist_by_neuralhash, hdic_confirm
from hdic.adapter import encode_hv

# ---------------- Helpers ---------------- #

def iter_images(root):
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir): continue
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.PNG"):
            for f in glob.glob(os.path.join(pdir, ext)):
                yield f, pid

# added helper to produce lists so we can show progress bars with totals
def list_images_all(root):
    if not root: return []
    items = []
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir): continue
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.PNG"):
            for f in glob.glob(os.path.join(pdir, ext)):
                items.append((f, pid))
    return sorted(items)

def run_probe(img_path, true_id, nh_db, hdic_db, Tnh, Thdic, K):
    """Cascade: NH shortlist -> HDIC best -> accept/reject."""
    img = load_and_align(img_path)
    if img is None:
        return {"skip": True}

    # Stage 1: NeuralHash shortlist
    q_bits = compute_hash_bits(img)
    shortlist, rejected = shortlist_by_neuralhash(q_bits, nh_db, Tnh, K)

    in_topk = any(pid == true_id for pid, _ in shortlist)
    if rejected or not shortlist:
        return {"skip": False, "in_topk": in_topk, "pred": None, "accepted": False, "best_d": np.inf}

    # Stage 2: HDIC confirmation among Top-K
    q_hv = encode_hv(img)
    best_d, best_pid = 1e9, None
    for pid, _ in shortlist:
        d, _ = hdic_confirm(q_hv, pid, hdic_db)
        if d < best_d:
            best_d, best_pid = d, pid

    accepted = (best_d <= Thdic)
    return {"skip": False, "in_topk": in_topk, "pred": best_pid, "accepted": accepted, "best_d": float(best_d)}

# ---------------- Metric sweep ---------------- #

def compute_far_frr_acc(known_root, unknown_root, nh_db, hdic_db, Tnh, Thdic, K):
    tot_known = rej_known = correct = 0
    known_items = list_images_all(known_root)
    for img_path, pid in tqdm(known_items, desc=f"Known (Tnh={Tnh},Th={Thdic})", unit="img", leave=False):
        out = run_probe(img_path, pid, nh_db, hdic_db, Tnh, Thdic, K)
        if out.get("skip"): continue
        tot_known += 1
        if out["accepted"] and out["pred"] == pid:
            correct += 1
        if not out["accepted"] or out["pred"] != pid:
            rej_known += 1
    frr = rej_known / (tot_known + 1e-9)
    acc = correct / (tot_known + 1e-9)

    tot_unk = fa = 0
    if unknown_root:
        unknown_items = list_images_all(unknown_root)
        for img_path, _ in tqdm(unknown_items, desc=f"Unknown (Tnh={Tnh},Th={Thdic})", unit="img", leave=False):
            out = run_probe(img_path, "__UNK__", nh_db, hdic_db, Tnh, Thdic, K)
            if out.get("skip"): continue
            tot_unk += 1
            if out["accepted"]: fa += 1
    far = fa / (tot_unk + 1e-9) if unknown_root else 0.0

    return far, frr, acc

def sweep_all(known_root, unknown_root, nh_db, hdic_db, Tnh_list, Thdic_grid, K, out_dir="plots", csv_path=None):
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    plt.figure()  # FAR/FRR vs Thdic
    for Tnh in Tnh_list:
        FARs, FRRs, ACCs, TARs, Ths = [], [], [], [], []
        # show progress over Thdic values
        for Thdic in tqdm(Thdic_grid, desc=f"Sweeping Thdic (Tnh={Tnh})", unit="th", leave=False):
            far, frr, acc = compute_far_frr_acc(known_root, unknown_root, nh_db, hdic_db, Tnh, Thdic, K)
            tar = 1 - frr
            FARs.append(far); FRRs.append(frr); ACCs.append(acc); TARs.append(tar); Ths.append(Thdic)
            all_results.append({"Tnh":Tnh,"Thdic":Thdic,"FAR":far,"FRR":frr,"TAR":tar,"Accuracy":acc})

        FARs, FRRs, ACCs, TARs, Ths = map(np.array, [FARs, FRRs, ACCs, TARs, Ths])

        # EER
        idx = np.argmin(np.abs(FARs - FRRs))
        eer = 0.5 * (FARs[idx] + FRRs[idx])
        print(f"[Tnh={Tnh}] EER={eer:.3f} at Thdic={Ths[idx]}")

        # Plot FAR/FRR
        plt.plot(Ths, FARs, label=f"FAR (Tnh={Tnh})")
        plt.plot(Ths, FRRs, linestyle="--", label=f"FRR (Tnh={Tnh})")
        plt.scatter([Ths[idx]], [eer], color="red")
        plt.text(Ths[idx], eer, f"EER={eer:.3f}", fontsize=7)

        # Accuracy vs Thdic
        plt.figure(2)
        plt.plot(Ths, ACCs, marker="o", label=f"Tnh={Tnh}")

        # ROC curve
        plt.figure(3)
        order = np.argsort(FARs)
        plt.plot(FARs[order], TARs[order], marker="o", label=f"Tnh={Tnh}, EER={eer:.3f}")

        # DIR vs FAR (open-set) → DIR = Accuracy on knowns
        if unknown_root:
            plt.figure(4)
            order = np.argsort(FARs)
            plt.plot(FARs[order], ACCs[order], marker="o", label=f"Tnh={Tnh}")


    # Save plots
    plt.figure(1); plt.xlabel("Thdic"); plt.ylabel("Rate"); plt.title("FAR & FRR vs Thdic"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "far_frr_vs_thdic.png"))

    plt.figure(2); plt.xlabel("Thdic"); plt.ylabel("Rank-1 Accuracy"); plt.title("Accuracy vs Thdic"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_vs_thdic.png"))

    plt.figure(3); plt.xlabel("FAR"); plt.ylabel("TAR"); plt.title("ROC Curve"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))

    if unknown_root:
        plt.figure(4); plt.xlabel("FAR"); plt.ylabel("DIR (Rank-1)"); plt.title("DIR vs FAR (Open-set)"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, "dir_vs_far.png"))

    if csv_path:
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"[SAVED] CSV → {csv_path}")

    print(f"[SAVED] plots in {out_dir}/")
    return all_results

# ---------------- CMC ---------------- #

def plot_cmc(known_root, nh_db, hdic_db, Tnh, Thdic, K, out_dir="plots"):
    ranks = []
    known_items = list_images_all(known_root)
    for img_path, pid in tqdm(known_items, desc=f"CMC probes (Tnh={Tnh},Th={Thdic})", unit="img"):
        out = run_probe(img_path, pid, nh_db, hdic_db, Tnh, Thdic, K)
        if out.get("skip"): continue
        if out["accepted"] and out["pred"] == pid:
            ranks.append(1)
        elif out["in_topk"]:
            ranks.append(K)  # at least in Top-K
        else:
            ranks.append(K+1)
    cmc = []
    N = len(ranks)
    for k in range(1, K+1):
        hits = sum(r <= k for r in ranks)
        cmc.append(hits / (N + 1e-9))
    plt.figure()
    plt.plot(range(1, K+1), cmc, marker="o")
    plt.xlabel("Rank"); plt.ylabel("Match Rate")
    plt.title(f"CMC Curve (Tnh={Tnh}, Thdic={Thdic})")
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"cmc_tnh{Tnh}_th{Thdic}.png"))
    print(f"[SAVED] CMC → {out_dir}/cmc_tnh{Tnh}_th{Thdic}.png")

# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="Closed-set probes")
    ap.add_argument("--unknown_root", default="", help="Open-set unknown probes")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--Tnh", type=str, default="30")
    ap.add_argument("--Thdic", type=str, default="2000:4000:200")
    ap.add_argument("--csv", default="plots/ident_results.csv")
    args = ap.parse_args()

    nh_db = load_nh_db()
    hdic_db = load_hdic_db()

    Tnh_list = [int(x) for x in args.Tnh.split(",")]
    if ":" in args.Thdic:
        a,b,c = args.Thdic.split(":")
        Thdic_grid = list(range(int(a), int(b)+1, int(c)))
    else:
        Thdic_grid = [int(x) for x in args.Thdic.split(",")]

    results = sweep_all(
        known_root=args.test_root,
        unknown_root=(args.unknown_root or None),
        nh_db=nh_db,
        hdic_db=hdic_db,
        Tnh_list=Tnh_list,
        Thdic_grid=Thdic_grid,
        K=args.K,
        out_dir="plots",
        csv_path=args.csv
    )

    # Plot CMC for the best operating point (first Tnh, mid Thdic)
    mid_th = Thdic_grid[len(Thdic_grid)//2]
    plot_cmc(args.test_root, nh_db, hdic_db, Tnh_list[0], mid_th, args.K, out_dir="plots")

if __name__ == "__main__":
    main()
