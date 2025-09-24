import os, glob, random, shutil
from pathlib import Path
from tqdm import tqdm

def list_images(p):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(p, e)))
    return sorted(files)

def split_person(person_dir, out_enroll, out_test, enroll_frac=0.7, seed=42, min_enroll=30, min_test=10, show_progress=True):
    random.seed(seed)
    imgs = list_images(person_dir)
    if not imgs:
        return 0,0
    random.shuffle(imgs)

    n = len(imgs)
    n_enroll = max(min_enroll, int(round(n * enroll_frac)))
    n_enroll = min(n_enroll, n - min_test) if n - min_test > 0 else n
    n_test = n - n_enroll

    pid = os.path.basename(person_dir.rstrip("/\\"))
    enroll_dir = os.path.join(out_enroll, pid)
    test_dir = os.path.join(out_test, pid)
    os.makedirs(enroll_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    enroll_list = imgs[:n_enroll]
    test_list = imgs[n_enroll:]

    if show_progress:
        # copy enroll images with progress
        for f in tqdm(enroll_list, desc=f"{pid} enroll", unit="img", leave=False):
            shutil.copy2(f, os.path.join(enroll_dir, os.path.basename(f)))
        for f in tqdm(test_list, desc=f"{pid} test", unit="img", leave=False):
            shutil.copy2(f, os.path.join(test_dir, os.path.basename(f)))
    else:
        for f in enroll_list:
            shutil.copy2(f, os.path.join(enroll_dir, os.path.basename(f)))
        for f in test_list:
            shutil.copy2(f, os.path.join(test_dir, os.path.basename(f)))

    return n_enroll, n_test

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root folder of original dataset (contains person folders)")
    ap.add_argument("--dst", required=True, help="Output root (will create enroll/ and test/)")
    ap.add_argument("--frac", type=float, default=0.7, help="Enroll fraction per identity (default 0.7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_enroll", type=int, default=30)
    ap.add_argument("--min_test", type=int, default=10)
    args = ap.parse_args()

    src = args.src
    out_enroll = os.path.join(args.dst, "enroll")
    out_test = os.path.join(args.dst, "test")
    os.makedirs(out_enroll, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    people = [p for p in glob.glob(os.path.join(src, "*")) if os.path.isdir(p)]
    total_enroll = total_test = 0

    for p in tqdm(people, desc="Persons", unit="person"):
        ne, nt = split_person(p, out_enroll, out_test, args.frac, args.seed, args.min_enroll, args.min_test, show_progress=True)
        total_enroll += ne
        total_test += nt
        # summary line per person shown after its internal progress bars
        print(f"{os.path.basename(p)} -> enroll {ne}, test {nt}")

    print(f"\nDONE. Total enroll={total_enroll}, test={total_test}")

if __name__ == "__main__":
    main()
