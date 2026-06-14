"""
ROC Curve — HDIC-only (parallel system, clustering prototypes, no NeuralHash).

Uses only what actually exists in the repo:
  - hdic/feature_extractor.py  → generate_embedding2(rgb_image)  (512-D FaceNet embedding)
  - hdic/encode_hv.py          → encode_embedding_to_hv(embedding) (10000-D binary HV)
  - common/hamming.py          → hamming_distance_bits()
  - preprocess/align.py        → load_and_align()
  - db/watchlist_hdic.jsonl    → loaded directly (no load_db function)

Run from repo root:
    python tools/plot_roc_hdic_only.py --dataset dataset/test --max_imgs 20
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from preprocess.align               import load_and_align
from hdic.feature_extractor         import generate_embedding2
from hdic.encode_hv                 import encode_embedding_to_hv
from common.hamming                 import hamming_distance_bits

# ── Constants ─────────────────────────────────────────────────────────────────
DIM_HV   = 10000
DB_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "watchlist_hdic.jsonl"
)


# ── Load HDIC DB directly from JSONL ─────────────────────────────────────────
def load_hdic_db(path: str) -> list:
    """Load db/watchlist_hdic.jsonl → list of {person_id, prototypes{cluster_N: [...]}}"""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # normalise key names across possible variants
            pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
            if pid and rec.get("prototypes"):
                records.append({"person_id": pid, "prototypes": rec["prototypes"]})
    return records


# ── Encode image → binary HV (full pipeline) ─────────────────────────────────
def image_to_hv(aligned_rgb: np.ndarray) -> np.ndarray:
    """
    aligned_rgb  →  generate_embedding2()  →  encode_embedding_to_hv()
    Returns (10000,) uint8 binary hypervector.
    """
    embedding = generate_embedding2(aligned_rgb)   # (512,) float32
    hv        = encode_embedding_to_hv(embedding)  # (10000,) uint8
    return hv


# ── HDIC score for one enrolled person (uses ALL cluster prototypes) ──────────
def hdic_score(probe_hv: np.ndarray, rec: dict) -> float:
    """
    Min Hamming distance over ALL cluster prototypes → similarity in [0,1].
    Identical logic to best_hdic_distance() in fusion/parallel.py.
    """
    best_dist = DIM_HV
    for _, proto in rec.get("prototypes", {}).items():
        proto_bits = np.array(proto, dtype=np.uint8)
        d = hamming_distance_bits(probe_hv, proto_bits)
        if d < best_dist:
            best_dist = d
    return 1.0 - (best_dist / DIM_HV)


# ── Image iterator with per-person cap ────────────────────────────────────────
def iter_images(root: str, max_per_person: int):
    """Yield (full_path, person_id), capped at max_per_person images each."""
    for person_dir in sorted(os.listdir(root)):
        full_dir = os.path.join(root, person_dir)
        if not os.path.isdir(full_dir):
            continue
        imgs = sorted([
            f for f in os.listdir(full_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])[:max_per_person]
        for fname in imgs:
            yield os.path.join(full_dir, fname), person_dir


# ── Collect genuine + impostor scores ────────────────────────────────────────
def collect_scores(dataset_root: str, hdic_db: list, max_imgs: int):
    pid_lookup = {rec["person_id"]: rec for rec in hdic_db}
    enrolled   = set(pid_lookup.keys())

    probe_list = [
        (p, pid) for p, pid in iter_images(dataset_root, max_imgs)
        if pid in enrolled
    ]

    print(f"\n[INFO] {len(probe_list)} probes  "
          f"({max_imgs} imgs/person × {len(enrolled)} enrolled persons)")
    print(f"[INFO] Each probe → 1 genuine + {len(hdic_db)-1} impostor score(s)")

    y_true, y_score = [], []

    for img_path, gt_pid in tqdm(probe_list, desc="Scoring", unit="img"):

        # Step 1: align
        aligned = load_and_align(img_path)
        if aligned is None:
            continue

        # Step 2: embed + encode HV
        try:
            probe_hv = image_to_hv(aligned)
        except Exception as e:
            print(f"  [WARN] Skipping {img_path}: {e}")
            continue

        # Step 3: genuine score — probe vs own prototypes
        y_true.append(1)
        y_score.append(hdic_score(probe_hv, pid_lookup[gt_pid]))

        # Step 4: impostor scores — probe vs every OTHER person's prototypes
        for rec in hdic_db:
            if rec["person_id"] == gt_pid:
                continue
            y_true.append(0)
            y_score.append(hdic_score(probe_hv, rec))

    return np.array(y_true, dtype=int), np.array(y_score, dtype=float)


# ── EER ───────────────────────────────────────────────────────────────────────
def compute_eer(fpr, tpr, thresholds):
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx]), idx


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ROC curve — HDIC only (no NeuralHash)"
    )
    parser.add_argument("--dataset",  required=True,
                        help="Test images root  e.g. dataset/test")
    parser.add_argument("--max_imgs", type=int, default=20,
                        help="Max images per person (default: 20)")
    args = parser.parse_args()

    # ── Load DB ───────────────────────────────────��───────────────────────────
    print(f"[INFO] Loading HDIC DB from {DB_PATH} ...")
    hdic_db = load_hdic_db(DB_PATH)
    if not hdic_db:
        print("[ERROR] HDIC DB is empty. Enroll people first.")
        return

    print(f"[INFO] Enrolled persons : {len(hdic_db)}")
    for rec in hdic_db:
        n_clusters = len(rec.get("prototypes", {}))
        print(f"       {rec['person_id']}  →  {n_clusters} cluster(s)")

    # ── Collect scores ────────────────────────────────────────────────────────
    y_true, y_score = collect_scores(args.dataset, hdic_db, args.max_imgs)

    if len(y_true) == 0:
        print("[ERROR] No scores collected. Check --dataset path.")
        return

    n_gen = int(y_true.sum())
    n_imp = int((y_true == 0).sum())
    print(f"\n[INFO] Genuine  scores : {n_gen}")
    print(f"[INFO] Impostor scores : {n_imp}")

    if n_gen == 0 or n_imp == 0:
        print("[ERROR] Need both genuine and impostor scores to plot ROC.")
        return

    # ── ROC ───────────────────────────────────────────────────────────────────
    fpr, tpr, thresholds     = roc_curve(y_true, y_score, pos_label=1)
    roc_auc                  = auc(fpr, tpr)
    eer, eer_thresh, eer_idx = compute_eer(fpr, tpr, thresholds)

    print(f"\n[RESULT] AUC = {roc_auc:.4f}")
    print(f"[RESULT] EER = {eer * 100:.2f}%  (threshold = {eer_thresh:.4f})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr,
            color='steelblue', linewidth=2.2,
            label=f'HDIC-only  (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1],
            color='grey', linewidth=1.2, linestyle='--',
            label='Random chance (AUC = 0.5)')

    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]],
               color='tomato', s=80, zorder=6,
               label=f'EER = {eer * 100:.2f}%')
    ax.annotate(
        f"EER = {eer * 100:.2f}%",
        xy=(fpr[eer_idx], tpr[eer_idx]),
        xytext=(fpr[eer_idx] + 0.06, tpr[eer_idx] - 0.09),
        fontsize=9, color='tomato',
        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2)
    )

    ax.set_xlabel('False Positive Rate (FPR)',  fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)',   fontsize=12)
    ax.set_title(
        'ROC Curve — HDIC Only\n'
        '(Parallel System · KMeans Clustering Prototypes · No NeuralHash)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roc_hdic_only.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()