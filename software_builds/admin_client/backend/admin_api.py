import os, json, yaml
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from tempfile import NamedTemporaryFile
import numpy as np

# ---- Import your real modules (unchanged) ----
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from hdic.feature_extractor import generate_embedding2
from hdic.encode_hv import encode_embedding_to_hv
from hdic.cluster_enroll import build_cluster_prototypes

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_DIR = REPO_ROOT / "db"
NH_FILE = DB_DIR / "watchlist_neuralhash.jsonl"
HDIC_FILE = DB_DIR / "watchlist_hdic.jsonl"
CFG_FILE = Path(__file__).resolve().parent / "config.yaml"

def ensure_files():
    DB_DIR.mkdir(exist_ok=True)
    if not NH_FILE.exists(): NH_FILE.write_text("")
    if not HDIC_FILE.exists(): HDIC_FILE.write_text("")
    if not CFG_FILE.exists():
        CFG_FILE.write_text(yaml.safe_dump(
            dict(Tnh=30, Thdic=3100, fused_th=0.7, w_nh=0.5, w_hdic=0.5)
        ))

def load_jsonl(path: Path):
    out = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s: out.append(json.loads(s))
    return out

def save_jsonl(path: Path, rows: list[dict]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)

def find_person(rows: list[dict], pid: str):
    for r in rows:
        if r.get("person_id") == pid:
            return r
    return None

@router.get("/list")
def list_persons():
    ensure_files()
    nh = load_jsonl(NH_FILE)
    hd = load_jsonl(HDIC_FILE)
    index = {}
    for r in nh:
        pid = r["person_id"]
        index.setdefault(pid, dict(name=r.get("name",""), nh_count=0, hdic_count=0))
        index[pid]["nh_count"] = len(r.get("hashes", []))
    for r in hd:
        pid = r["person_id"]
        index.setdefault(pid, dict(name=r.get("name",""), nh_count=0, hdic_count=0))
        index[pid]["hdic_count"] = len(r.get("prototypes", {}))
    return dict(persons=[
        dict(person_id=pid, name=info.get("name",""),
             nh_count=info.get("nh_count",0), hdic_count=info.get("hdic_count",0))
        for pid, info in sorted(index.items())
    ])

@router.get("/view/{person_id}")
def view_person(person_id: str):
    ensure_files()
    nh = load_jsonl(NH_FILE)
    hd = load_jsonl(HDIC_FILE)
    nhr = find_person(nh, person_id)
    hdr = find_person(hd, person_id)
    if not nhr and not hdr:
        raise HTTPException(status_code=404, detail="Person not found")
    return dict(
        person_id=person_id,
        name=(nhr or hdr).get("name",""),
        nh_hash_count=len(nhr.get("hashes", [])) if nhr else 0,
        hdic_proto_keys=list(hdr.get("prototypes", {}).keys()) if hdr else [],
    )

@router.post("/enroll")
async def enroll_person(
    person_id: str = Form(...),
    name: str = Form(""),
    files: List[UploadFile] = File(...),
):
    """
    Enroll or add images:
    - NH: append 96-bit 0/1 arrays to 'hashes'
    - HDIC: append binary HVs to 'prototypes' as p{idx}
    """
    ensure_files()
    nh = load_jsonl(NH_FILE)
    hd = load_jsonl(HDIC_FILE)
    nhr = find_person(nh, person_id)
    hdr = find_person(hd, person_id)
    if nhr is None:
        nhr = {"person_id": person_id, "name": name, "hashes": []}
        nh.append(nhr)
    if hdr is None:
        hdr = {"person_id": person_id, "name": name, "prototypes": {}}
        hd.append(hdr)

    added = 0
    failed = 0
    embeddings = []
    added = 0
    failed = 0

    for _idx, uf in enumerate(files):
        tmp_path = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await uf.read())
                tmp_path = tmp.name

            face = load_and_align(tmp_path, output_size=(160,160), normalize=False)
            if face is None:
                failed += 1
                continue

            # === NeuralHash processing ===
            nh_bits = compute_hash_bits(face)
            nhr["hashes"].append(np.array(nh_bits, dtype=np.uint8).tolist())

            # === HDIC Embedding Collection ===
            emb = generate_embedding2(face)
            embeddings.append(emb)

            added += 1
        except Exception as e:
            print("[ERROR] Enrollment failed for one image:", e)
            failed += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # === Perform HDIC clustering ===
    if len(embeddings) > 0:
        # Encode all embeddings → HVs (10k-D binary)
        hvs = [encode_embedding_to_hv(e) for e in embeddings]

        # Cluster into groups (same as your original HDIC)
        try:
            prototypes = build_cluster_prototypes(hvs, num_clusters=3)
            hdr["prototypes"] = {k: v.tolist() for k, v in prototypes.items()}
        except Exception as e:
            print("[ERROR] Clustering failed:", e)
            hdr["prototypes"] = {"cluster_0": hvs[0].tolist()}

    # === Save everything ===
    save_jsonl(NH_FILE, nh)
    save_jsonl(HDIC_FILE, hd)

    return dict(
        status="ok",
        person_id=person_id,
        added_images=added,
        failed_images=failed,
        total=len(files),
        clusters=len(hdr["prototypes"])
    )


@router.delete("/delete/{person_id}")
def delete_person(person_id: str):
    ensure_files()
    nh = load_jsonl(NH_FILE)
    hd = load_jsonl(HDIC_FILE)
    nh2 = [r for r in nh if r.get("person_id") != person_id]
    hd2 = [r for r in hd if r.get("person_id") != person_id]
    save_jsonl(NH_FILE, nh2)
    save_jsonl(HDIC_FILE, hd2)
    return dict(status="deleted", person_id=person_id)

@router.get("/config")
def get_config():
    ensure_files()
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@router.post("/config")
def update_config(cfg: dict):
    ensure_files()
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return dict(status="updated", config=cfg)
