import numpy as np
from .settings import NH_DIR, HD_DIR
from .bitpack import pack_probe_uint64, hamming_rows_all
from .packed_store import PackedStore

# import your repo modules directly
from preprocess.align import align_from_array
from neuralhash.adapter import compute_hash_bits
from hdic.feature_extractor import generate_embedding2
from hdic.encode_hv import encode_embedding_to_hv

def _reduce_min_per_person(dists: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.minimum.reduceat(dists, offsets[:-1])

class PackedMatcher:
    def __init__(self):
        self.reload()

    def reload(self):
        nh = PackedStore(NH_DIR, bits=96)
        hd = PackedStore(HD_DIR, bits=10000)
        self.NH_ROWS, self.NH_OFFS, self.NH_PIDS = nh.load_memmap()
        self.HD_ROWS, self.HD_OFFS, self.HD_PIDS = hd.load_memmap()
        # Align person order
        hd_idx = {pid: i for i, pid in enumerate(self.HD_PIDS)}
        self.COMMON = [pid for pid in self.NH_PIDS if pid in hd_idx]
        self.NH_IDX = np.asarray([self.NH_PIDS.index(pid) for pid in self.COMMON], dtype=np.int32)
        self.HD_IDX = np.asarray([hd_idx[pid] for pid in self.COMMON], dtype=np.int32)

    def match_frame(self, frame_bgr: np.ndarray,
                    Tnh: float, Thdic: float, w_nh: float, w_hdic: float, fused_th: float):
        face = align_from_array(frame_bgr, output_size=(160,160), normalize=False)
        if face is None:
            return {"decision": "NO_FACE", "person_id": None, "scores": {}}

        nh_bits = compute_hash_bits(face).astype(np.uint8).ravel()
        emb     = generate_embedding2(face)
        hv_bits = encode_embedding_to_hv(emb).astype(np.uint8).ravel()

        if not self.COMMON or self.NH_ROWS.shape[0]==0 or self.HD_ROWS.shape[0]==0:
            return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}

        p_nh = pack_probe_uint64(nh_bits)
        p_hd = pack_probe_uint64(hv_bits)

        d_all_nh = hamming_rows_all(p_nh, self.NH_ROWS)
        d_all_hd = hamming_rows_all(p_hd, self.HD_ROWS)

        dmin_nh = _reduce_min_per_person(d_all_nh, self.NH_OFFS)[self.NH_IDX]
        dmin_hd = _reduce_min_per_person(d_all_hd, self.HD_OFFS)[self.HD_IDX]

        Snh  = 1.0 - (dmin_nh / 96.0)
        Shd  = 1.0 - (dmin_hd / 10000.0)
        Sfin = (w_nh * Snh) + (w_hdic * Shd)

        k = int(np.argmax(Sfin))
        pid = self.COMMON[k]
        scores = {
            "d_nh": int(dmin_nh[k]),
            "d_hdic": int(dmin_hd[k]),
            "Snh": float(Snh[k]),
            "Shdic_norm": float(Shd[k]),
            "Sfinal": float(Sfin[k]),
        }
        is_match = (scores["d_nh"] < Tnh) and (scores["d_hdic"] < Thdic) and (scores["Sfinal"] >= fused_th)
        return {"decision": "MATCH" if is_match else "NO_MATCH", "person_id": pid, "scores": scores}
