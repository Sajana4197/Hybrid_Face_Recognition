import argparse
import glob
import os
from tqdm import tqdm

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from hdic.adapter import encode_hv
from hdic.cluster_enroll import build_cluster_prototypes
from neuralhash.db import enroll_person as nh_enroll
from hdic.db import enroll_person as hdic_enroll

def expand_glob(pattern):
    if os.path.isdir(pattern):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(pattern, ext)))
        return sorted(files)
    else:
        return sorted(glob.glob(pattern))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Person ID")
    parser.add_argument("--name", required=True, help="Person name")
    parser.add_argument("--images", required=True, help="Glob pattern for images")
    args = parser.parse_args()

    images = expand_glob(args.images)
    if not images:
        print(f"[ERROR] No images found for pattern: {args.images}")
        return

    nh_hashes = []
    hv_list = []
    used, skipped = 0, 0

    t = tqdm(images, desc="Processing images", unit="img")
    for p in t:
        img = load_and_align(p)
        if img is None:
            skipped += 1
            t.set_postfix(file=os.path.basename(p), used=used, skipped=skipped)
            t.write(f"[WARN] Skipping {p} (no valid face)")
            continue

        try:
            # NeuralHash 96-bit
            hbits = compute_hash_bits(img)
            nh_hashes.append(hbits.tolist())

            # HDIC hypervector
            hv = encode_hv(img)
            hv_list.append(hv)

            used += 1
            t.set_postfix(file=os.path.basename(p), used=used, skipped=skipped)
        except Exception as e:
            skipped += 1
            t.set_postfix(file=os.path.basename(p), used=used, skipped=skipped)
            t.write(f"[ERROR] Failed processing {p}: {e}")
            continue

    t.close()

    if not nh_hashes or not hv_list:
        print("[ERROR] No valid samples processed, enrollment aborted.")
        return

    print(f"Enrolling NeuralHash ({len(nh_hashes)} samples)...")
    pbar_nh = tqdm(total=1, desc="NeuralHash enroll")
    nh_enroll(args.id, args.name, nh_hashes)
    pbar_nh.update(1)
    pbar_nh.close()
    print("NeuralHash enrollment finished")

    print("Building and enrolling HDIC prototypes...")
    cluster_protos = build_cluster_prototypes(hv_list, num_clusters=3)
    pbar_hd = tqdm(total=1, desc="HDIC enroll")
    hdic_enroll(args.id, args.name, cluster_protos)
    pbar_hd.update(1)
    pbar_hd.close()
    print("HDIC enrollment finished")

    print(f"[DONE] Enrollment complete for {args.name} ({args.id})")
    print(f"Images processed: {len(images)} | Used: {used} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
