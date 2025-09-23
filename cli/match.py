import argparse
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from fusion.cascade import shortlist_by_neuralhash, hdic_confirm
from hdic.adapter import encode_hv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to probe image")
    parser.add_argument("--K", type=int, default=5, help="NeuralHash shortlist size")
    parser.add_argument("--Tnh", type=int, default=30, help="NeuralHash reject threshold")
    parser.add_argument("--Thdic", type=int, default=5000, help="HDIC reject threshold")
    args = parser.parse_args()

    # Step 1: Preprocess
    probe = load_and_align(args.image)
    if probe is None:
        print(f"[ERROR] No valid face detected in probe image: {args.image}")
        return

    # Step 2: NeuralHash
    probe_bits = compute_hash_bits(probe)

    # Step 3: Load DBs
    nh_db = load_nh_db()
    hdic_db = load_hdic_db()
    if not nh_db or not hdic_db:
        print("[ERROR] One or both databases are empty. Please enroll first.")
        return

    # Step 4: NeuralHash shortlist
    shortlist, rejected = shortlist_by_neuralhash(probe_bits, nh_db, args.Tnh, args.K)

    print(f"\n[INFO] NeuralHash shortlist (Top-{args.K}):")
    for pid, dist in shortlist:
        print(f"  - {pid} → Hamming distance = {dist}")

    if rejected:
        print("[REJECT] Probe rejected at NeuralHash stage (distance too high)")
        return

    # Step 5: HDIC confirmation
    probe_hv = encode_hv(probe)
    best_person = None
    best_dist = 1e9

    for pid, _ in shortlist:
        dist, cluster = hdic_confirm(probe_hv, pid, hdic_db)
        print(f"[INFO] HDIC check → {pid}, cluster={cluster}, distance={dist}")
        if dist < best_dist:
            best_dist = dist
            best_person = pid

    # Step 6: Final decision
    if best_person and best_dist < args.Thdic:
        print(f"\n[MATCH] {best_person} (HDIC distance={best_dist:.3f})")
    else:
        print("\n[REJECT] Probe did not match any enrolled person")

if __name__ == "__main__":
    main()
