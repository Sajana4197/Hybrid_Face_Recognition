# db/packed_store.py
import json
import os
from pathlib import Path
import numpy as np
from fusion.bitpack import pack_bits_uint64

class PackedStore:
    """
    Append-only, person-contiguous store for binary hashes/hypervectors.

    On disk under base_dir:
      - data.npy          : uint64 matrix (R, W)  — packed rows
      - offsets.npy       : int64 array  (P+1,)   — start indices per person (prefix sums)
      - person_ids.json   : list[str] length P
      - person_names.json : list[str] length P
      - meta.json         : {"bits": int, "words": int}
    """

    def __init__(self, base_dir: Path, bits: int):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.bits = int(bits)
        self.words = (self.bits + 63) // 64

        self.data_path    = self.base_dir / "data.npy"
        self.offsets_path = self.base_dir / "offsets.npy"
        self.pids_path    = self.base_dir / "person_ids.json"
        self.pnames_path  = self.base_dir / "person_names.json"
        self.meta_path    = self.base_dir / "meta.json"

        self._ensure_files()

    # ---------------------------
    # Internal helpers (atomic I/O)
    # ---------------------------
    def _atomic_save_npy(self, path: Path, arr: np.ndarray):
        """
        Atomic write for .npy files on Windows:
        - write exact bytes to a temp file opened in 'wb' (prevents .npy auto-suffix)
        - fsync, then replace
        """
        tmp = path.with_suffix(path.suffix + ".tmp")  # e.g., data.npy.tmp
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            # ensure no extra .npy added by numpy
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _atomic_save_json(self, path: Path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)

    def _load_npy(self, path: Path):
        # Standard np.load with allow_pickle=False for safety
        return np.load(path, allow_pickle=False)

    # ---------------------------
    # Bootstrap / validation
    # ---------------------------
    def _ensure_files(self):
        # meta
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("bits") != self.bits or meta.get("words") != self.words:
                raise ValueError(
                    f"Meta mismatch for store at {self.base_dir}: "
                    f"expected bits/words {self.bits}/{self.words}, "
                    f"found {meta.get('bits')}/{meta.get('words')}"
                )
        else:
            self._atomic_save_json(self.meta_path, {"bits": self.bits, "words": self.words})

        # data.npy
        if not self.data_path.exists():
            empty = np.zeros((0, self.words), dtype=np.uint64)
            self._atomic_save_npy(self.data_path, empty)

        # offsets.npy
        if not self.offsets_path.exists():
            self._atomic_save_npy(self.offsets_path, np.array([0], dtype=np.int64))

        # person_ids.json / person_names.json
        if not self.pids_path.exists():
            self._atomic_save_json(self.pids_path, [])
        if not self.pnames_path.exists():
            self._atomic_save_json(self.pnames_path, [])

    # ---------------------------
    # Write path
    # ---------------------------
    def append_person(self, person_id: str, person_name: str, samples_bits01: np.ndarray):
        """
        Append a NEW person block.

        samples_bits01: (N, bits) uint8 in {0,1}
        Policy: no in-place append to an existing pid. If needed, rebuild externally.
        """
        # Validate input
        if samples_bits01.ndim != 2 or samples_bits01.shape[1] != self.bits:
            raise ValueError(
                f"samples_bits01 must be (N,{self.bits}) binary; got {samples_bits01.shape}"
            )

        # Load current state
        data = self._load_npy(self.data_path)              # (R, W) uint64
        offsets = self._load_npy(self.offsets_path)        # (P+1,) int64
        pids = json.loads(self.pids_path.read_text(encoding="utf-8"))
        pnames = json.loads(self.pnames_path.read_text(encoding="utf-8"))

        if person_id in pids:
            raise ValueError(f"Person '{person_id}' already exists in packed store.")

        # Pack and append
        samples_bits01 = samples_bits01.astype(np.uint8, copy=False)
        packed = pack_bits_uint64(samples_bits01)          # (N, W) uint64

        new_data = packed if data.size == 0 else np.concatenate([data, packed], axis=0)
        new_offsets = np.concatenate([offsets, [offsets[-1] + packed.shape[0]]]).astype(np.int64, copy=False)

        pids.append(person_id)
        pnames.append(person_name)

        # Atomic writes (order: data -> offsets -> ids -> names)
        self._atomic_save_npy(self.data_path, new_data)
        self._atomic_save_npy(self.offsets_path, new_offsets)
        self._atomic_save_json(self.pids_path, pids)
        self._atomic_save_json(self.pnames_path, pnames)

    # ---------------------------
    # Read path (inference-friendly)
    # ---------------------------
    def load_memmap(self):
        """
        Return memory-mapped arrays for fast, low-RAM reads:
          data   : (R, W) uint64 (memmap, read-only)
          offsets: (P+1,) int64 (memmap, read-only)
          pids   : list[str] length P
        """
        data = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
        pids = json.loads(self.pids_path.read_text(encoding="utf-8"))
        return data, offsets, pids
