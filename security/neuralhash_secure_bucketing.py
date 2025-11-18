# neuralhash_secure_bucketing.py

import os
import hmac
import hashlib
from typing import Iterable, List, Tuple, Union

# ===============================
# Bit helpers
# ===============================

def hex96_to_bits(hex_str: str) -> List[int]:
    """
    Convert a 96-bit hex string (24 hex chars) to a list of 96 bits (0/1), big-endian per byte.
    Accepts optional '0x' prefix.
    """
    s = hex_str.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != 24:
        raise ValueError(f"Expected 24 hex chars (96 bits), got {len(s)}")
    b = bytes.fromhex(s)
    if len(b) != 12:
        raise ValueError("Expected 12 bytes")
    bits: List[int] = []
    for byte in b:
        for bit in range(7, -1, -1):
            bits.append((byte >> bit) & 1)
    return bits  # length 96

def bits96_to_hex(bits: Iterable[int]) -> str:
    """
    Convert 96 bits (0/1) to a 24-char hex string (big-endian per byte).
    """
    bits_list = [1 if int(x) else 0 for x in bits]
    if len(bits_list) != 96:
        raise ValueError(f"Expected 96 bits, got {len(bits_list)}")
    out = bytearray()
    for i in range(0, 96, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits_list[i + j]
        out.append(byte)
    return out.hex()

def split_96bits_into_16x6(values: Iterable[int]) -> List[int]:
    """
    Given 96 bits as an iterable of 0/1, split into 16 buckets of 6 bits each.
    Returns list of 16 integers in [0..63]. Bit order is left-to-right in the given iterable.
    """
    bits = [1 if int(x) else 0 for x in values]
    if len(bits) != 96:
        raise ValueError(f"Expected 96 bits, got {len(bits)}")
    buckets: List[int] = []
    for i in range(16):
        v = 0
        base = 6 * i
        for j in range(6):
            v = (v << 1) | bits[base + j]
        buckets.append(v)  # 0..63
    return buckets

# ===============================
# Keyed 6-bit -> 40-bit mapping
# ===============================

_DOMAIN = b"NH6->40:v1"  # domain separation / versioning

def map_6bit_to_40bit(
    six_bit_value: int,
    bucket_index: int,
    key: bytes,
    user_salt: bytes = b"",
) -> bytes:
    """
    Deterministically map one 6-bit value (0..63) to 40 bits (5 bytes) using HMAC-SHA256.
    Includes domain, bucket index, 6-bit value, and optional user_salt in the MAC input.

    Returns: 5 bytes.
    """
    if not (0 <= six_bit_value <= 63):
        raise ValueError("six_bit_value must be in [0, 63]")
    if not (0 <= bucket_index <= 15):
        raise ValueError("bucket_index must be in [0, 15]")
    if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
        raise ValueError("key must be bytes with sufficient entropy (>=16 bytes recommended; 32 bytes ideal).")
    msg = _DOMAIN + b"|" + bytes([bucket_index]) + bytes([six_bit_value]) + user_salt
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return digest[:5]  # 40 bits

def map_all_buckets_to_40bit_tokens(
    buckets: Iterable[int],
    key: bytes,
    user_salt: bytes = b"",
) -> List[bytes]:
    """
    Map 16 bucket values (each 0..63) to 16 tokens of 5 bytes each.
    """
    b_list = list(buckets)
    if len(b_list) != 16:
        raise ValueError(f"Expected 16 buckets, got {len(b_list)}")
    return [map_6bit_to_40bit(v, i, key, user_salt=user_salt) for i, v in enumerate(b_list)]

def tokens_bytes_to_hex(tokens: Iterable[bytes]) -> List[str]:
    """
    Convert list of 5-byte tokens to list of 10-char hex strings.
    """
    out = []
    for t in tokens:
        if len(t) != 5:
            raise ValueError("Each token must be 5 bytes")
        out.append(t.hex())
    return out

def tokens_hex_to_bytes(tokens_hex: Iterable[str]) -> List[bytes]:
    """
    Convert list of 10-char hex strings to list of 5-byte tokens.
    """
    out = []
    for hx in tokens_hex:
        s = hx.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        if len(s) != 10:
            raise ValueError("Each token hex must have 10 hex characters (40 bits).")
        b = bytes.fromhex(s)
        if len(b) != 5:
            raise ValueError("Each token must be 5 bytes")
        out.append(b)
    return out

# ===============================
# Enrollment and matching
# ===============================

def enroll_from_bits(
    bits96: Iterable[int],
    key: bytes,
    user_salt: bytes = b"",
) -> List[bytes]:
    """
    From raw 96 bits, produce 16 obfuscated 40-bit tokens (bytes).
    """
    buckets = split_96bits_into_16x6(bits96)
    tokens = map_all_buckets_to_40bit_tokens(buckets, key=key, user_salt=user_salt)
    return tokens  # list of 16 items, each 5 bytes

def enroll_from_hex96(
    hex96: str,
    key: bytes,
    user_salt: bytes = b"",
    as_hex: bool = True,
) -> Union[List[bytes], List[str]]:
    """
    From a 96-bit hex string (24 hex chars), produce 16 obfuscated 40-bit tokens.
    If as_hex=True, return list of 10-char hex strings; else return bytes.
    """
    bits = hex96_to_bits(hex96)
    tokens = enroll_from_bits(bits, key=key, user_salt=user_salt)
    return tokens_bytes_to_hex(tokens) if as_hex else tokens

def match_tokens(
    probe_tokens: Iterable[bytes],
    enroll_tokens: Iterable[bytes],
    threshold: int,
) -> Tuple[bool, int]:
    """
    Compare probe tokens and enrolled tokens bucket-by-bucket (same index).
    Returns (is_match, matches_count).
    """
    p = list(probe_tokens)
    e = list(enroll_tokens)
    if len(p) != 16 or len(e) != 16:
        raise ValueError("Need 16 tokens on both sides")
    matches = sum(1 for i in range(16) if p[i] == e[i])
    return (matches >= threshold, matches)

def match_tokens_hex(
    probe_tokens_hex: Iterable[str],
    enroll_tokens_hex: Iterable[str],
    threshold: int,
) -> Tuple[bool, int]:
    """
    Hex-string version of match_tokens.
    """
    p = tokens_hex_to_bytes(probe_tokens_hex)
    e = tokens_hex_to_bytes(enroll_tokens_hex)
    return match_tokens(p, e, threshold)

# ===============================
# Threshold selection guidance
# ===============================
def recommended_threshold_for_hamming(h_max: int, mode: str = "worst") -> int:
    """
    Recommend a bucket-match threshold given a desired maximum tolerated Hamming distance h_max
    over the 96-bit original hash.

    - mode="worst": assumes bit errors spread across distinct buckets;
      threshold = 16 - min(h_max, 16)
    - mode="best": assumes errors concentrate within as few buckets as possible;
      threshold = 16 - ceil(h_max / 6)

    Returns an integer in [0..16].
    """
    h = max(0, int(h_max))
    if mode == "worst":
        return max(0, 16 - min(h, 16))
    elif mode == "best":
        from math import ceil
        return max(0, 16 - min(16, ceil(h / 6)))
    else:
        raise ValueError("mode must be 'worst' or 'best'")

# ===============================
# Optional: key loading helper
# ===============================
def load_key_from_env(var: str = "NH_BUCKET_KEY") -> bytes:
    """
    Load a device-secret key from environment. If absent, raises.
    """
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing environment variable {var} for bucketing key")
    # Support hex-encoded or raw text
    try:
        key = bytes.fromhex(val) if all(c in "0123456789abcdefABCDEF" for c in val.replace("0x", "")) and len(val.strip().replace("0x","")) % 2 == 0 else val.encode("utf-8")
    except Exception:
        key = val.encode("utf-8")
    if len(key) < 16:
        raise RuntimeError(f"{var} must provide >= 16 bytes of key material; 32 bytes recommended.")
    return key