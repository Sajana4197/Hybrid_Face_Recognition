import argparse
import os
from typing import List

# Import your existing modules (adjust package-relative imports if needed)
try:
    from neuralhash_api import compute_neural_hash_hex_from_path
except ImportError:
    # If your files are in a package, use: from yourpkg.neuralhash_api import ...
    raise

try:
    from neuralhash_secure_bucketing import (
        enroll_from_hex96,
        tokens_bytes_to_hex,
        match_tokens_hex,
        load_key_from_env,
        recommended_threshold_for_hamming,
    )
except ImportError:
    # If your files are in a package, use: from yourpkg.neuralhash_secure_bucketing import ...
    raise


def tokens_hex_from_image(image_path: str, key: bytes, user_salt: bytes) -> List[str]:
    # 1) Compute 96-bit NeuralHash (24 hex chars)
    hex96 = compute_neural_hash_hex_from_path(image_path)
    # 2) Convert to 16 obfuscated 40-bit tokens (each 10 hex chars)
    tokens_hex = enroll_from_hex96(hex96, key=key, user_salt=user_salt, as_hex=True)
    return tokens_hex


def main():
    p = argparse.ArgumentParser(description="Compute bucketed 40-bit tokens from an image and optionally match.")
    p.add_argument("image", help="Path to the input image")
    p.add_argument("--probe", help="Optional second image path to compare against the first")
    p.add_argument("--user-salt", default="", help="Per-user stable salt (string). Kept stable for the same identity.")
    p.add_argument("--key-hex", default="", help="Optional hex-encoded device key override. If not provided, reads NH_BUCKET_KEY env.")
    p.add_argument("--hamming-max", type=int, default=10, help="Desired maximum tolerated Hamming distance for matching guidance.")
    args = p.parse_args()

    # Load key (prefer --key-hex, else env NH_BUCKET_KEY)
    if args.key_hex:
        key = bytes.fromhex(args.key_hex)
    else:
        key = load_key_from_env("NH_BUCKET_KEY")

    user_salt = args.user_salt.encode("utf-8")

    # Enrollment for the first image
    enrolled_tokens_hex = tokens_hex_from_image(args.image, key=key, user_salt=user_salt)
    print("Enrolled tokens (16 x 40-bit hex):")
    print(enrolled_tokens_hex)

    # Optional: match a probe image
    if args.probe:
        probe_tokens_hex = tokens_hex_from_image(args.probe, key=key, user_salt=user_salt)

        # Pick a threshold; start with conservative "worst" mode
        threshold = recommended_threshold_for_hamming(args.hamming_max, mode="worst")
        is_match, matches = match_tokens_hex(probe_tokens_hex, enrolled_tokens_hex, threshold)
        print(f"Probe tokens: {probe_tokens_hex}")
        print(f"Threshold (worst-case for h_max={args.hamming_max}): {threshold}")
        print(f"Bucket matches: {matches} / 16")
        print(f"Match result: {is_match}")


if __name__ == "__main__":
    main()