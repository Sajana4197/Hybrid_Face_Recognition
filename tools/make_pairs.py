import os, glob, csv, random

def list_persons(root):
    return sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])

def list_images(path):
    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return sorted(imgs)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="dataset/test")
    ap.add_argument("--out_csv", required=True, help="pairs.csv")
    ap.add_argument("--impostor_ratio", type=float, default=1.0, help="impostor:genuine ratio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    persons = list_persons(args.test_root)
    pid2imgs = {os.path.basename(p): list_images(p) for p in persons}
    rows = []

    # Genuine: every test image matched against its own ID
    for pid, imgs in pid2imgs.items():
        for img in imgs:
            rows.append([img, "genuine", pid])

    # Impostors: for each person, sample some other person's ID
    pids = list(pid2imgs.keys())
    impostor_rows = []
    for pid, imgs in pid2imgs.items():
        other_pids = [x for x in pids if x != pid]
        for img in imgs:
            if random.random() < args.impostor_ratio:
                neg_pid = random.choice(other_pids)
                impostor_rows.append([img, "impostor", neg_pid])

    rows.extend(impostor_rows)
    random.shuffle(rows)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path","label","target_person_id"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
