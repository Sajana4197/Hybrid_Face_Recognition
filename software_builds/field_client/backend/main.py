import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json, yaml, uvicorn, cv2, numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

# ---- Internal project imports ----
from fusion.parallel_service import match_frame
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from hdic.feature_extractor import generate_embedding2
from hdic.encode_hv import encode_embedding_to_hv
from software_builds.field_client.backend.loader import load_watchlists
from software_builds.field_client.backend.matcher import score_person_distances, fuse_parallel

# ---- DB imports (Neon / SQLAlchemy) ----
# Requires db.py, models.py, crud.py as provided earlier
from sqlalchemy.orm import Session
from .db import init_db, get_db
from . import crud

# ----- FastAPI app -----
app = FastAPI(title="Hybrid Field Client (Parallel NH+HDIC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----- Config loading -----
CFG_PATH = Path(__file__).parent / "config.yaml"
CONFIG = yaml.safe_load(CFG_PATH.read_text())
Tnh     = float(CONFIG.get("Tnh", 30))
Thdic   = float(CONFIG.get("Thdic", 3100))
w_nh    = float(CONFIG.get("w_nh", 0.5))
w_hdic  = float(CONFIG.get("w_hdic", 0.5))
T_final = float(CONFIG.get("T_final", 0.7))
PORT    = int(CONFIG.get("port", 5001))
LOGPATH = Path(__file__).parent / (CONFIG.get("log_path", "logs/matches.jsonl"))
LOGPATH.parent.mkdir(parents=True, exist_ok=True)

# Load merged watchlist (NH + HDIC)
REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONS = load_watchlists(REPO_ROOT)

# ----- Startup: init DB schema -----
@app.on_event("startup")
def on_startup():
    # Create tables if they don't exist (against Neon if DATABASE_URL is set)
    init_db()

# ---------- Utility ----------
def log_match(entry: dict):
    with LOGPATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "watchlist_size": len(PERSONS)}

# ---------- Config endpoints ----------
@app.get("/config")
def get_config():
    return {"Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": w_hdic, "T_final": T_final}

@app.post("/config")
def update_config(payload: dict):
    global Tnh, Thdic, w_nh, w_hdic, T_final
    for k in ["Tnh","Thdic","w_nh","w_hdic","T_final"]:
        if k in payload:
            val = float(payload[k])
            if k in ("w_nh","w_hdic") and not (0.0 <= val <= 1.0):
                continue
            locals()[k] = val
    cfg = {"Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": w_hdic,
           "T_final": T_final, "port": PORT, "log_path": str(LOGPATH.relative_to(Path(__file__).parent))}
    CFG_PATH.write_text(yaml.safe_dump(cfg))
    return {"ok": True, "config": cfg}

# ---------- Enrollment (stores embedding in Neon) ----------
@app.post("/enroll")
async def enroll(
    person_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Enroll a new person by uploading a face image.
    - Aligns the face
    - Generates an embedding
    - Stores the embedding in the cloud DB (Neon) via SQLAlchemy
    """
    # Read and decode image
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Align and validate face
    try:
        aligned = load_and_align(frame)  # expected BGR/RGB aligned face image
        if aligned is None:
            raise ValueError("No face detected")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Face alignment failed: {e}")

    # Generate embedding (HDIC/FaceNet)
    try:
        embedding_vec = generate_embedding2(aligned)  # typically numpy array
        if embedding_vec is None:
            raise ValueError("Failed to generate embedding")
        # Convert to JSON-serializable list of floats
        embedding_list = np.asarray(embedding_vec, dtype=np.float32).flatten().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    # Store in DB
    rec = crud.create_face_embedding(
        db=db,
        person_name=person_name,
        embedding=embedding_list,
        image_path=None,  # optionally store to object storage and save URL here
    )
    return {"ok": True, "id": rec.id, "person_name": rec.person_name}

# ---------- Single-frame match ----------
@app.post("/match")
async def match(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), dtype=np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        return {"decision": "ERROR", "error": "Invalid image"}

    result = match_frame(
        frame_bgr=frame,
        Tnh=Tnh, Thdic=Thdic,
        w_nh=w_nh, w_hdic=w_hdic,
        fused_th=T_final
    )
    result["frames"] = 1
    log_match(result)
    return result

# ---------- Multi-frame match (stable memory version) ----------
@app.post("/match_multi")
async def match_multi(files: List[UploadFile] = File(...)):
    """
    Capture & match using multiple frames (e.g., 5 webcam images).
    Each frame is matched independently; final decision = majority rule (>=3 MATCH).
    """
    frames_data = [(i, await uf.read(), uf.filename) for i, uf in enumerate(files)]

    def process_single(idx, img_bytes, filename):
        try:
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Could not decode frame {filename}")
                return None

            result = match_frame(
                frame_bgr=frame,
                Tnh=Tnh,
                Thdic=Thdic,
                w_nh=w_nh,
                w_hdic=w_hdic,
                fused_th=T_final,
            )

            Sfinal = result.get("scores", {}).get("Sfinal", 0)
            d_nh = result.get("scores", {}).get("d_nh", None)
            d_hdic = result.get("scores", {}).get("d_hdic", None)
            decision = result.get("decision", "UNKNOWN")
            pid = result.get("person_id", None)

            print(f"[INFO] Frame {idx+1}: Decision={decision}, Sfinal={float(Sfinal):.3f}")

            return {
                "index": idx + 1,
                "filename": filename,
                "decision": decision,
                "Sfinal": round(float(Sfinal), 3) if Sfinal is not None else None,
                "d_nh": d_nh,
                "d_hdic": d_hdic,
                "person_id": pid,
            }
        except Exception as e:
            print(f"[WARN] Failed to process frame {idx+1}: {e}")
            return None

    # Run all frames in parallel
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(filter(None, ex.map(lambda args: process_single(*args), frames_data)))

    # ---- Handle no valid results ----
    if not results:
        print("[WARN] No valid frame results found — returning NO_FACE")
        return {"decision": "NO_FACE", "frames": 0}

    # ---- Majority-based rule ----
    total = len(results)
    match_count = sum(1 for r in results if r["decision"] == "MATCH")
    majority_decision = "MATCH" if match_count >= 3 else "NO_MATCH"

    # ---- Best frame (highest Sfinal) ----
    best_frame = max(results, key=lambda x: x.get("Sfinal", 0) or 0.0)
    best_score = best_frame.get("Sfinal", 0.0)
    best_pid = best_frame.get("person_id", None)

    # ---- Final combined result ----
    final_result = {
        "method": "majority-of-N",
        "frames": total,
        "match_frames": match_count,
        "match_ratio": round(match_count / total, 2),
        "decision": majority_decision,
        "best_score": round(float(best_score), 3),
        "best_person_id": best_pid,
        "frame_details": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    log_match(final_result)
    return final_result

# ---------- Entry ----------
if __name__ == "__main__":
    uvicorn.run(
        "software_builds.field_client.backend.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=True,
    )