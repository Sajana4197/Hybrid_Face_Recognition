import os
import sys
import argparse
from typing import Iterable, Tuple, Union

import numpy as np

# Support running both as a package module and as a standalone script
try:
    from .neuralhash_core import (
        load_pca,
        load_hyperplanes,
        get_96d_neural_hash_from_array,
    )
except ImportError:
    from neuralhash_core import (  # type: ignore
        load_pca,
        load_hyperplanes,
        get_96d_neural_hash_from_array,
    )

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required to load images. Install with: pip install Pillow") from e


# ---------------- Load global assets ---------------- #

ASSETS_DIR = os.path.join("./neuralhash/assets")
PCA_PATH = os.path.join(ASSETS_DIR, "pca_512_to_128.pkl")
DAT_PATH = os.path.join(ASSETS_DIR, "neuralhash_128x96_seed1.dat")

if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(f"PCA file not found: {PCA_PATH}")
if not os.path.exists(DAT_PATH):
    raise FileNotFoundError(f"NeuralHash .dat file not found: {DAT_PATH}")

PCA_MODEL = load_pca(PCA_PATH)
HYPERPLANES = load_hyperplanes(DAT_PATH)


# ---------------- Core hash functions ---------------- #

def compute_hash_bits(rgb_image: np.ndarray) -> np.ndarray:
    """
    Compute the 96-bit NeuralHash for an aligned RGB image array.

    Input:
        rgb_image: numpy array (160x160x3, dtype=uint8 or float in [0,1])
    Output:
        numpy array of shape (96,), dtype=uint8 (0/1 bits)
    """
    hbits = get_96d_neural_hash_from_array(rgb_image, PCA_MODEL, HYPERPLANES)
    if hbits.shape[0] != 96:
        raise ValueError(f"Unexpected NeuralHash output shape: {hbits.shape}")
    # Ensure dtype is uint8 with 0/1 values
    hbits = (hbits.astype(np.uint8) & 1).reshape(-1)
    return hbits


def bits_to_bytes(bits: np.ndarray, bitorder: str = "big") -> bytes:
    """
    Pack 96 bits into 12 bytes.
    bitorder = 'big' means the first bit is the MSB of the first byte.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.shape[0] != 96:
        raise ValueError(f"Expected 96 bits, got {bits.shape[0]}")
    return np.packbits(bits, bitorder=bitorder).tobytes()


def bits_to_hex(bits: np.ndarray, bitorder: str = "big", with_prefix: bool = False) -> str:
    """
    Return a 24-hex-character string for the 96-bit hash.
    """
    hx = bits_to_bytes(bits, bitorder=bitorder).hex()
    return ("0x" + hx) if with_prefix else hx


# ---------------- Image helpers ---------------- #

def load_image_as_rgb_array(
    image_path: str,
    size: Tuple[int, int] = (160, 160),
    keep_as_float: bool = False,
) -> np.ndarray:
    """
    Load an image file, convert to RGB, resize to the model's expected size, and return an array.

    Args:
        image_path: Path to the input image.
        size: (width, height) to resize to. Default: (160, 160).
        keep_as_float: If True, returns float32 in [0, 1]. Otherwise uint8 [0, 255].

    Returns:
        numpy array of shape (H, W, 3)
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize(size, resample=Image.BILINEAR)
    arr = np.asarray(im)
    if keep_as_float:
        arr = (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        arr = arr.astype(np.uint8)
    return arr


# ---------------- High-level convenience APIs ---------------- #

def compute_neural_hash_bits_from_path(image_path: str) -> np.ndarray:
    """
    Compute 96-bit NeuralHash from an image on disk. Returns bits as np.uint8 array of shape (96,).
    """
    rgb = load_image_as_rgb_array(image_path, size=(160, 160), keep_as_float=False)
    return compute_hash_bits(rgb)


def compute_neural_hash_hex_from_path(image_path: str, with_prefix: bool = False) -> str:
    """
    Compute 96-bit NeuralHash from an image on disk. Returns a 24-hex-character string (optionally '0x' prefixed).
    """
    bits = compute_neural_hash_bits_from_path(image_path)
    return bits_to_hex(bits, bitorder="big", with_prefix=with_prefix)


def compute_neural_hash_bytes_from_path(image_path: str) -> bytes:
    """
    Compute 96-bit NeuralHash from an image on disk. Returns 12 raw bytes.
    """
    bits = compute_neural_hash_bits_from_path(image_path)
    return bits_to_bytes(bits, bitorder="big")


# ---------------- CLI ---------------- #

def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute 96-bit NeuralHash for input images."
    )
    p.add_argument("images", nargs="+", help="Path(s) to input image(s)")
    p.add_argument(
        "--format",
        choices=["hex", "bits", "bytes"],
        default="hex",
        help="Output format. 'hex' prints 24 hex chars per image, "
             "'bits' prints 96 bits (0/1), 'bytes' writes 12 raw bytes to stdout per image.",
    )
    p.add_argument(
        "--prefix",
        action="store_true",
        help="With --format hex, prefix hex output with '0x'."
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if ns.format == "bytes" and len(ns.images) > 1 and sys.stdout.isatty():
            print("Refusing to print binary for multiple images to a terminal. Redirect to a file or use --format hex/bits.", file=sys.stderr)
            return 2

        for path in ns.images:
            if ns.format == "hex":
                hx = compute_neural_hash_hex_from_path(path, with_prefix=ns.prefix)
                print(f"{path}\t{hx}")
            elif ns.format == "bits":
                bits = compute_neural_hash_bits_from_path(path).astype(np.uint8)
                bitstr = "".join(str(int(b)) for b in bits.tolist())
                print(f"{path}\t{bitstr}")
            else:  # bytes
                data = compute_neural_hash_bytes_from_path(path)
                # For single image write raw bytes to stdout; for multiple, print path + hex to avoid corrupting terminal
                if len(ns.images) == 1 and not sys.stdout.isatty():
                    sys.stdout.buffer.write(data)
                else:
                    print(f"{path}\t{data.hex()}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())