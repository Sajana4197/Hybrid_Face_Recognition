# fusion/bitpack.py
import numpy as np

# ---------------------------------------------------------
# Packing: {0,1} bits -> uint64 words
# ---------------------------------------------------------
def pack_bits_uint64(bits: np.ndarray) -> np.ndarray:
    """
    bits: (N, M) uint8 in {0,1}
    returns: (N, W) uint64, where W = ceil(M/64)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2:
        raise ValueError(f"bits must be 2D (N,M); got {bits.shape}")

    N, M = bits.shape
    W = (M + 63) // 64
    out = np.zeros((N, W), dtype=np.uint64)
    # pack 64 columns at a time
    for w in range(W):
        lo = w * 64
        hi = min(lo + 64, M)
        chunk = bits[:, lo:hi].astype(np.uint64, copy=False)  # (N, <=64)
        # set bits by shifting each column j
        for j in range(hi - lo):
            out[:, w] |= (chunk[:, j] << j)
    return out

def pack_probe_uint64(probe_bits: np.ndarray) -> np.ndarray:
    """
    probe_bits: (M,) uint8 in {0,1}
    returns: (W,) uint64
    """
    b = np.asarray(probe_bits, dtype=np.uint8).ravel()
    M = b.size
    W = (M + 63) // 64
    out = np.zeros((W,), dtype=np.uint64)
    for i in range(M):
        if b[i]:
            out[i // 64] |= (np.uint64(1) << (i % 64))
    return out

# ---------------------------------------------------------
# Popcount (bitcount) — portable fallback if np.bit_count is missing
# ---------------------------------------------------------
_has_np_bitcount = hasattr(np, "bit_count")

if _has_np_bitcount:
    def _popcount_words(u64: np.ndarray) -> np.ndarray:
        # u64: (..., W) uint64 -> (..., W) uint64 counts per word
        return np.bit_count(u64)
else:
    # uint8 lookup table: number of set bits in [0..255]
    _POPC_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount_words(u64: np.ndarray) -> np.ndarray:
        """
        u64: (..., W) uint64
        returns: (..., W) uint32 counts per word (sum of 8 bytes per word)
        """
        # reinterpret each uint64 word as 8 uint8 bytes
        u8 = u64.view(np.uint8)            # shape (..., W*8)
        # reshape last axis to (..., W, 8) then look up & sum
        new_shape = u64.shape + (8,)
        u8 = u8.reshape(new_shape)
        # LUT sum across the 8 bytes to get popcount per word
        return _POPC_LUT[u8].sum(axis=-1, dtype=np.uint32)

# ---------------------------------------------------------
# Hamming distance to all rows
# ---------------------------------------------------------
def hamming_rows_all(probe_u64: np.ndarray, db_u64: np.ndarray) -> np.ndarray:
    """
    probe_u64: (W,)  uint64
    db_u64   : (R,W) uint64
    returns  : (R,)  int32   (Hamming distance per row)
    """
    if db_u64.ndim != 2:
        raise ValueError(f"db_u64 must be 2D (R,W); got {db_u64.shape}")
    if probe_u64.ndim != 1:
        raise ValueError(f"probe_u64 must be 1D (W,); got {probe_u64.shape}")
    if db_u64.shape[1] != probe_u64.shape[0]:
        raise ValueError(f"W mismatch: db has {db_u64.shape[1]} words, probe has {probe_u64.shape[0]}")

    xor = np.bitwise_xor(db_u64, probe_u64)    # (R,W)
    pcw = _popcount_words(xor)                 # (R,W)
    return pcw.sum(axis=1).astype(np.int32)
