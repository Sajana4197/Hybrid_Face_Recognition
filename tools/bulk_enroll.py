import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to enroll dataset root (e.g., dataset/enroll)")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERROR] {root} is not a valid directory")
        sys.exit(1)

    persons = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if not persons:
        print(f"[ERROR] No person folders found under {root}")
        sys.exit(1)

    for pid in persons:
        folder = os.path.join(root, pid)
        print(f"[INFO] Enrolling {pid} from {folder} ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "cli.enroll", "--id", pid, "--name", pid, "--images", folder],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Enrollment failed for {pid}: {e}")

    print("\n[DONE] Bulk enrollment complete.")

if __name__ == "__main__":
    main()
