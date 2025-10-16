import os
import sys
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np

# Local imports (adjust if in a package)
try:
    from neuralhash_api import compute_hash_bits
except ImportError:
    from neuralhash_api import compute_hash_bits  # type: ignore

try:
    from neuralhash_secure_bucketing import (
        enroll_from_bits,           # returns 16 tokens (bytes each, 5 bytes)
        load_key_from_env,
        recommended_threshold_for_hamming,
    )
except ImportError:
    from neuralhash_secure_bucketing import enroll_from_bits, load_key_from_env, recommended_threshold_for_hamming  # type: ignore

from db import fetch_all_templates

def detect_and_crop_face_bgr(bgr: np.ndarray, target_size: int = 160) -> np.ndarray:
    """
    Detects the largest face using Haar cascade, crops with some margin, resizes to (160, 160).
    If no face detected, falls back to center crop and resize.
    Returns an RGB uint8 array of shape (160, 160, 3).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))

    h, w = gray.shape
    if len(faces) == 0:
        # Fallback: center square crop
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = bgr[y0:y0+side, x0:x0+side]
    else:
        # Pick largest area face
        x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])
        # Expand box slightly
        cx, cy = x + fw // 2, y + fh // 2
        side = int(max(fw, fh) * 1.3)
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        crop = bgr[y0:y1, x0:x1]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.uint8)

def chunk_tokens80_to_list(tokens80: bytes) -> List[bytes]:
    if len(tokens80) != 80:
        raise ValueError("tokens80 must be 80 bytes")
    return [tokens80[i*5:(i+1)*5] for i in range(16)]

def count_bucket_matches(tokens_a: List[bytes], tokens_b: List[bytes]) -> int:
    return sum(1 for i in range(16) if tokens_a[i] == tokens_b[i])

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Capture from webcam, compute tokens, and match against DB.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--hamming-max", type=int, default=10, help="Max tolerated Hamming distance guidance")
    p.add_argument("--salt", default=os.getenv("NH_SALT", ""), help="Global salt (string). Must match enrollment.")
    p.add_argument("--topk", type=int, default=5, help="Show top-K matches by bucket score")
    args = p.parse_args(argv)

    key = load_key_from_env()
    user_salt = args.salt.encode("utf-8")

    # Load templates from DB
    rows = fetch_all_templates()
    if not rows:
        print("No templates in DB. Run enrollment first.")
        return 2

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera.")
        return 3

    print("Press SPACE to capture, ESC to quit.")
    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        view = frame.copy()
        cv2.putText(view, "Press SPACE to capture, ESC to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live", view)
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return 0
        if keycode == 32:  # SPACE
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame is None:
        print("No frame captured.")
        return 4

    # Face crop -> 160x160 RGB
    rgb = detect_and_crop_face_bgr(frame, target_size=160)

    # Compute 96-bit hash bits, then map to tokens with same key+salt as enrollment
    bits96 = compute_hash_bits(rgb)
    probe_tokens = enroll_from_bits(bits96, key=key, user_salt=user_salt)  # list of 16 x 5-byte tokens

    # Compare against DB
    scored = []
    for r in rows:
        cand_tokens = chunk_tokens80_to_list(r["tokens_80"])
        matches = count_bucket_matches(probe_tokens, cand_tokens)
        scored.append((matches, r["username"], str(r["person_id"]), str(r["template_id"])))

    scored.sort(key=lambda x: x[0], reverse=True)

    threshold = recommended_threshold_for_hamming(args.hamming_max, mode="worst")
    print(f"Threshold (worst-case for h_max={args.hamming_max}): {threshold}")

    # Show top-K
    print("Top matches:")
    for i, (m, uname, pid, tid) in enumerate(scored[:args.topk]):
        print(f"{i+1:>2}. {uname}  matches={m}/16  person_id={pid}  template_id={tid}")

    best_m, best_uname, _, _ = scored[0]
    is_match = best_m >= threshold
    print(f"\nBest match: {best_uname}  ({best_m}/16)  -> {'ACCEPT' if is_match else 'REJECT'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())