import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import sys, json, yaml, uvicorn, cv2, numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from fusion.parallel_service import match_frame

# ----- Project wiring -----
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../<repo-root>/
sys.path.append(str(REPO_ROOT))

# Import YOUR real modules (used in the CLI parallel system)
from preprocess.align import load_and_align                 # use the same aligner your CLI uses
from neuralhash.adapter import compute_hash_bits       # returns 96-bit vector of 0/1
from hdic.encode_hv import encode_embedding_to_hv                    # returns 10k-D float32 hypervector
# If you have helper fusion code, keep using it; here we compute Sfinal inline.

from software_builds.field_client.backend.loader import load_watchlists
from software_builds.field_client.backend.matcher import score_person_distances, fuse_parallel

# ----- FastAPI app -----
app = FastAPI(title="Hybrid Field Client (Parallel NH+HDIC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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

# Load merged watchlist (requires both JSONL files under repo/db/)
PERSONS = load_watchlists(REPO_ROOT)

@app.get("/health")
def health():
    return {"status": "ok", "watchlist_size": len(PERSONS)}

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
    # persist
    cfg = {"Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": w_hdic, "T_final": T_final, "port": PORT, "log_path": str(LOGPATH.relative_to(Path(__file__).parent))}
    CFG_PATH.write_text(yaml.safe_dump(cfg))
    return {"ok": True, "config": cfg}

def detect_and_align(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Use your real aligner. If your aligner requires RGB or a specific API, adapt here.
    """
    try:
        # load_and_align should return a cropped/aligned face image (np.ndarray, BGR or RGB as your encoders expect)
        face = load_and_align(frame_bgr)
        return face
    except Exception:
        # Fallback: Haar cascade (only for emergency)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x,y,w,h = faces[0]
        return frame_bgr[y:y+h, x:x+w]

def log_match(entry: dict):
    with LOGPATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
    return result

if __name__ == "__main__":
    uvicorn.run("software_builds.field_client.backend.main:app", host="127.0.0.1", port=PORT, reload=True)
