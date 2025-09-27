
import argparse
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from hdic.adapter import encode_hv
from fusion.parallel import best_neuralhash_distance, best_hdic_distance, decide_parallel

def main():
    parser = argparse.ArgumentParser(description="Parallel fusion matcher (NeuralHash + HDIC)")
    parser.add_argument("--image", required=True, help="Path to probe image")
    parser.add_argument("--Tnh", type=int, default=25, help="NeuralHash reject threshold (distance)")
    parser.add_argument("--Thdic", type=int, default=3000, help="HDIC accept threshold (distance)")
    parser.add_argument("--w_nh", type=float, default=0.5, help="Weight for NeuralHash similarity")
    parser.add_argument("--w_hdic", type=float, default=0.5, help="Weight for HDIC similarity")
    parser.add_argument("--fused_th", type=float, default=0.7, help="Fused similarity threshold [0..1]")
    parser.add_argument("--require_both", action="store_true", help="Require both modalities to pass their thresholds")
    args = parser.parse_args()

    # Step 1: Preprocess
    aligned = load_and_align(args.image)

    # Step 2: Compute both signatures
    probe_hash = compute_hash_bits(aligned)          # (96,) uint8
    probe_hv   = encode_hv(aligned)                  # (10000,) uint8

    # Step 3: Load watchlists
    nh_db  = load_nh_db()
    hd_db  = load_hdic_db()

    # Step 4: Compute distances per person for both modalities
    nh_d   = best_neuralhash_distance(probe_hash, nh_db)
    hd_d   = best_hdic_distance(probe_hv, hd_db)

    # Step 5: Decide
    best_pid, debug = decide_parallel(
        nh_d, hd_d,
        Tnh_reject=args.Tnh,
        Thdic_accept=args.Thdic,
        w_nh=args.w_nh,
        w_hdic=args.w_hdic,
        fused_threshold=args.fused_th,
        require_both_modalities=args.require_both,
    )

    # Step 6: Report
    print("\n[DETAILS] per-person scores:")
    for pid, info in sorted(debug.items(), key=lambda x: -x[1]["fused"])[:5]:
        print(f"  {pid}: nh_d={info['nh_d']}, hdic_d={info['hdic_d']}, "
            f"nh_sim={info['nh_sim']:.3f}, hdic_sim={info['hdic_sim']:.3f}, "
            f"fused={info['fused']:.3f}")

    if best_pid is not None:
        print(f"\n[MATCH] {best_pid} (fused >= {args.fused_th})")
    else:
        print("\n[REJECT] Probe did not match any enrolled person (parallel).")


if __name__ == "__main__":
    main()
