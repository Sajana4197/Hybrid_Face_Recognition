# enroll_dataset.py

import os
import argparse
import glob
from typing import List, Tuple

import numpy as np

# Local imports (adjust if in a package)
try:
    from neuralhash_api import compute_neural_hash_hex_from_path
except ImportError:
    from neuralhash_api import compute_neural_hash_hex_from_path  # type: ignore

try:
    from neuralhash_secure_bucketing import (
        enroll_from_hex96,   # returns 16 tokens
        load_key_from_env,
        _DOMAIN,             # version/tag string used by mapping
    )
except ImportError:
    from neuralhash_secure_bucketing import enroll_from_hex96, load_key_from_env, _DOMAIN  # type: ignore

from db import init_db, upsert_person, insert_template

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_persons_and_images(dataset_root: str, recursive: bool = True) -> List[Tuple[str, List[str]]]:
    """
    Expect structure:
      dataset_root/
        personA/*.jpg|png
        personB/*.jpg|png
    Returns: [(person_name, [image_paths...]), ...]
    """
    persons = []
    for entry in os.scandir(dataset_root):
        if entry.is_dir():
            person = entry.name
            pattern = os.path.join(entry.path, "**", "*.*") if recursive else os.path.join(entry.path, "*.*")
            files = [p for p in glob.glob(pattern, recursive=recursive)
                     if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
            if files:
                persons.append((person, sorted(files)))
    return sorted(persons, key=lambda x: x[0].lower())

def tokens_hex_to_80bytes(tokens_hex: List[str]) -> bytes:
    """
    Convert 16 token hex strings (10 hex chars each) into a single 80-byte blob.
    """
    if len(tokens_hex) != 16:
        raise ValueError("Expected 16 tokens")
    parts = []
    for hx in tokens_hex:
        s = hx.lower().strip()
        if s.startswith("0x"):
            s = s[2:]
        if len(s) != 10:
            raise ValueError("Each token must be 40 bits (10 hex chars)")
        parts.append(bytes.fromhex(s))
    blob = b"".join(parts)
    if len(blob) != 80:
        raise ValueError("Internal error: expected 80 bytes")
    return blob

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Enroll dataset into cloud Postgres using 16x40-bit tokens per image.")
    p.add_argument("--dataset-root", required=True, help="Path to dataset root with subfolders per person.")
    p.add_argument("--init-db", action="store_true", help="Create tables if not present.")
    p.add_argument("--per-person", choices=["all", "first"], default="all",
                   help="Enroll all images per person (default) or only the first image.")
    p.add_argument("--salt", default=os.getenv("NH_SALT", ""), help="Global salt (string). Same value must be used for matching.")
    args = p.parse_args(argv)

    if args.init_db:
        init_db()

    key = load_key_from_env()  # reads NH_BUCKET_KEY
    user_salt = args.salt.encode("utf-8")

    persons = list_persons_and_images(args.dataset_root, recursive=True)
    if not persons:
        print("No persons/images found. Check dataset structure.")
        return 2

    total_templates = 0
    for person_name, images in persons:
        # Use a global/deployment salt for 1:N identification so tokens are comparable across users
        pid = upsert_person(person_name, user_salt=user_salt)

        # FIXED: argparse uses per_person for --per-person
        img_list = images if args.per_person == "all" else images[:1]
        for img_path in img_list:
            try:
                hex96 = compute_neural_hash_hex_from_path(img_path)
                tokens_hex = enroll_from_hex96(hex96, key=key, user_salt=user_salt, as_hex=True)
                tokens_blob = tokens_hex_to_80bytes(tokens_hex)
                insert_template(pid, tokens_blob, version=_DOMAIN.decode() if isinstance(_DOMAIN, bytes) else _DOMAIN)
                total_templates += 1
                print(f"Enrolled: {person_name} <- {os.path.relpath(img_path, args.dataset_root)}")
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    print(f"Done. Persons: {len(persons)}, Templates stored: {total_templates}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())