import os
from typing import List, Dict, Any
import numpy as np
from common.io_utils import append_jsonl, read_jsonl

DB_PATH = "db/watchlist_neuralhash.jsonl"

def enroll_person(person_id: str, name: str, hashes) -> None:
    """
    Enroll a person into the NeuralHash watchlist.
    hashes: list of numpy arrays or Python lists, each of length 96
    """
    hash_lists = []
    for h in hashes:
        if isinstance(h, np.ndarray):
            hash_lists.append(h.astype(int).tolist())
        elif isinstance(h, list):
            hash_lists.append([int(x) for x in h])  # ensure ints
        else:
            raise TypeError(f"Unsupported hash type: {type(h)}")

    rec = {
        "person_id": person_id,
        "name": name,
        "hashes": hash_lists
    }
    append_jsonl(DB_PATH, rec)


def load_db() -> List[Dict[str, Any]]:
    """
    Load the NeuralHash watchlist database.
    Returns a list of records:
        {
          "person_id": str,
          "name": str,
          "hashes": [[0,1,0,...,1], ...]  # each of length 96
        }
    """
    return read_jsonl(DB_PATH)
