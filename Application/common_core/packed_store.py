import json, os
from pathlib import Path
import numpy as np
from .bitpack import pack_bits_uint64

class PackedStore:
    """
    Append-only store for binary hashes/hypervectors.
    base_dir/
      data.npy         : uint64 (R, W)
      offsets.npy      : int64  (P+1,)
      person_ids.json  : list[str]
      person_names.json: list[str]
      meta.json        : {"bits": int, "words": int}
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

    def _ensure_files(self):
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if (meta.get("bits") != self.bits) or (meta.get("words") != self.words):
                raise ValueError(f"PackedStore meta mismatch: {meta} vs {self.bits}/{self.words}")
        else:
            self.meta_path.write_text(json.dumps({"bits": self.bits, "words": self.words}))
        if not self.data_path.exists():
            np.save(self.data_path, np.zeros((0, self.words), dtype=np.uint64))
        if not self.offsets_path.exists():
            np.save(self.offsets_path, np.array([0], dtype=np.int64))
        if not self.pids_path.exists():
            self.pids_path.write_text(json.dumps([]))
        if not self.pnames_path.exists():
            self.pnames_path.write_text(json.dumps([]))

    def _atomic_save_npy(self, path: Path, arr: np.ndarray):
        tmp = Path(str(path) + ".tmp.npy")
        np.save(tmp, arr, allow_pickle=False)
        os.replace(tmp, path)

    def _atomic_save_json(self, path: Path, obj):
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)

    def append_person(self, person_id: str, person_name: str, samples_bits01: np.ndarray):
        data    = np.load(self.data_path, allow_pickle=False)
        offsets = np.load(self.offsets_path, allow_pickle=False)
        pids    = json.loads(self.pids_path.read_text(encoding="utf-8"))
        pnames  = json.loads(self.pnames_path.read_text(encoding="utf-8"))

        if person_id in pids:
            raise ValueError(f"Person '{person_id}' already exists")

        packed = pack_bits_uint64(samples_bits01.astype(np.uint8, copy=False))
        new_data    = packed if data.size == 0 else np.concatenate([data, packed], axis=0)
        new_offsets = np.concatenate([offsets, [offsets[-1] + packed.shape[0]]])

        pids.append(person_id); pnames.append(person_name)

        self._atomic_save_npy(self.data_path, new_data)
        self._atomic_save_npy(self.offsets_path, new_offsets)
        self._atomic_save_json(self.pids_path, pids)
        self._atomic_save_json(self.pnames_path, pnames)

    def load_memmap(self):
        data    = np.load(self.data_path, mmap_mode="r")
        offsets = np.load(self.offsets_path, mmap_mode="r")
        pids    = json.loads(self.pids_path.read_text(encoding="utf-8"))
        return data, offsets, pids
