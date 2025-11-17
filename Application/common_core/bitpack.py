import numpy as np

def pack_bits_uint64(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2:
        raise ValueError("pack_bits_uint64 expects (N,M) array of {0,1}")
    N, M = bits.shape
    W = (M + 63) // 64
    out = np.zeros((N, W), dtype=np.uint64)
    for w in range(W):
        lo, hi = w*64, min((w+1)*64, M)
        chunk = bits[:, lo:hi].astype(np.uint64, copy=False)
        for j in range(hi - lo):
            out[:, w] |= (chunk[:, j] << j)
    return out

def pack_probe_uint64(probe_bits: np.ndarray) -> np.ndarray:
    b = np.asarray(probe_bits, dtype=np.uint8).ravel()
    W = (b.size + 63) // 64
    out = np.zeros((W,), dtype=np.uint64)
    for i, bit in enumerate(b):
        if bit:
            out[i // 64] |= (np.uint64(1) << (i % 64))
    return out

_has_np_bitcount = hasattr(np, "bit_count")
if _has_np_bitcount:
    def _popcount_words(u64: np.ndarray) -> np.ndarray:
        return np.bit_count(u64)
else:
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    def _popcount_words(u64: np.ndarray) -> np.ndarray:
        u8 = u64.view(np.uint8).reshape(u64.shape + (8,))
        return _LUT[u8].sum(axis=-1, dtype=np.uint32)

def hamming_rows_all(probe_u64: np.ndarray, db_u64: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(db_u64, probe_u64)   # (R,W)
    pcw = _popcount_words(xor)                # (R,W)
    return pcw.sum(axis=1).astype(np.int32)
