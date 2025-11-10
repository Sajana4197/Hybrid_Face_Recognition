# tools/bulk_enroll_packed.py
import os, argparse, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to dataset root with per-person folders")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERROR] {root} is not a valid directory")
        sys.exit(1)

    persons = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if not persons:
        print(f"[ERROR] no person folders under {root}")
        sys.exit(1)

    for pid in sorted(persons):
        folder = os.path.join(root, pid)
        print(f"[INFO] enrolling {pid} from {folder}")
        try:
            subprocess.run(
                [sys.executable, "-m", "cli.enroll_packed",
                 "--id", pid, "--name", pid, "--images", folder],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] failed {pid}: {e}")

    print("\n[DONE] bulk enrollment complete")

if __name__ == "__main__":
    main()
