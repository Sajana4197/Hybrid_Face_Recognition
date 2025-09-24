import os
from typing import List, Dict, Any
import numpy as np
from common.io_utils import append_jsonl, read_jsonl

DB_PATH = "db/watchlist_hdic.jsonl"

def enroll_person(person_id: str, name: str, prototypes: Dict[str, np.ndarray]) -> None:
    """
    Enroll a person into the HDIC watchlist.

    Args:
        person_id: unique identifier (e.g. "n000001")
        name: person's name
        prototypes: dict { "cluster_x": numpy array (10000,), dtype=uint8 }
    """
    proto_serialized = {cid: hv.astype(int).tolist() for cid, hv in prototypes.items()}

    rec: Dict[str, Any] = {
        "person_id": person_id,
        "name": name,
        "prototypes": proto_serialized  # no image paths
    }
    append_jsonl(DB_PATH, rec)

def load_db() -> List[Dict[str, Any]]:
    """
    Load the HDIC watchlist database.
    Returns a list of records:
        {
          "person_id": str,
          "name": str,
          "prototypes": {
              "cluster_0": [0,1,0,...],
              "cluster_1": [1,0,1,...],
              ...
          }
        }
    """
    return read_jsonl(DB_PATH)
