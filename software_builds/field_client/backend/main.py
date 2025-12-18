import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import requests
import json, yaml, uvicorn, cv2, numpy as np
import threading
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from math import ceil
from software_builds.field_client.backend.verifications import router as ver_router, enqueue_case

# ---- Internal project imports ----
from fusion.parallel_service import match_frame
from software_builds.field_client.backend.loader import load_watchlists
from software_builds.field_client.backend.led_controller import led

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

last_led_state = None

# ----- FastAPI app -----
app = FastAPI(title="Hybrid Field Client (Parallel NH+HDIC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ver_router)

# ----- Config loading -----
CFG_PATH = Path(__file__).parent / "config.yaml"
CONFIG = yaml.safe_load(CFG_PATH.read_text())
Tnh     = float(CONFIG.get("Tnh", 30))
Thdic   = float(CONFIG.get("Thdic", 3100))
w_nh    = float(CONFIG.get("w_nh", 0.2))
w_hdic  = float(CONFIG.get("w_hdic", 0.8))
T_final = float(CONFIG.get("T_final", 0.8))
PORT    = int(CONFIG.get("port", 5001))
LOGPATH = Path(__file__).parent / (CONFIG.get("log_path", "logs/matches.jsonl"))
LOGPATH.parent.mkdir(parents=True, exist_ok=True)

# NEW: Load Admin API address from config.yaml
ADMIN_API = CONFIG.get("admin_api", "http://127.0.0.1:5002")

# Load merged watchlist (NH + HDIC)
REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONS = {}

def reload_watchlist():
    """
    Reload watchlist from NPZ cache ONLY (super fast). 
    Does NOT rebuild cache - admin does that.
    """
    global PERSONS
    
    print("\n" + "="*60)
    print("[RELOAD] Fast reload from NPZ cache")
    print("="*60)
    
    # Reload fusion service data from NPZ cache (FAST!)
    from fusion import parallel_service
    count = parallel_service.load_watchlist_data()
    
    # Build PERSONS dict from the fusion engine data (fast, no file I/O)
    PERSONS = {}
    for pid in parallel_service. PERSON_IDS:
        name = parallel_service.NAME_MAP. get(pid, pid)
        PERSONS[pid] = {
            "name": name,
            "nh_hashes": parallel_service.NH_MAP.get(pid, []),
            "hdic_prototypes": parallel_service.HD_MAP.get(pid, [])
        }
    
    print(f"[INFO] ✅ Reload complete in <1 second!")
    print(f"[INFO]    Loaded {count} persons from NPZ cache")
    print("="*60 + "\n")
    
    return count

# Initial load
reload_watchlist()

def rebuild_cache_async():
    """Rebuild cache in background thread (non-blocking)"""
    def _rebuild():
        try:
            from db.build_cache import build_cache
            print("[INFO] 🔄 Rebuilding cache in background...")
            build_cache(silent=True)
            print("[INFO] ✅ Cache rebuild complete")
        except Exception as e:
            print(f"[ERROR] Cache rebuild failed: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=_rebuild, daemon=True)
    thread.start()
    return thread


def rebuild_cache_sync():
    """Rebuild cache synchronously (blocking)"""
    try:
        from db.build_cache import build_cache
        print("[INFO] 🔄 Rebuilding cache synchronously...")
        build_cache(silent=False)
        print("[INFO] ✅ Cache rebuild complete")
        return True
    except Exception as e:
        print(f"[ERROR] Cache rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---------- Utility ----------
def log_match(entry:  dict):
    with LOGPATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def send_manual_alert(person_id: str, score: float, image_bytes: bytes, timestamp: str):
    """
    Send captured best-match frame to Admin for manual verification.
    Includes person ID, score, and timestamp.
    """
    try:
        files = {"file": ("match.jpg", BytesIO(image_bytes), "image/jpeg")}
        data = {"person_id": person_id, "score": score, "timestamp": timestamp}
        r = requests.post(
            f"{ADMIN_API}/manual_check/receive",
            files=files,
            data=data,
            timeout=8,
        )
        print(f"[INFO] Sent manual verification alert: {r.status_code}")
    except Exception as e:
        print("[WARN] Failed to send manual verification alert:", e)


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "watchlist_size": len(PERSONS)}

# ---------- Reload watchlist ----------
@app.post("/reload")
def reload_watchlist_endpoint():
    """Rebuild cache and reload the watchlist (FULL REBUILD)"""
    import time
    start = time.time()
    
    try:
        print("\n" + "="*60)
        print("[ENDPOINT] /reload called - Starting full rebuild")
        print("="*60)
        
        # Step 1: Rebuild cache synchronously (blocking, but thorough)
        success = rebuild_cache_sync()
        
        if not success:
            raise RuntimeError("Cache rebuild failed - check logs for details")
        
        # Step 2: Reload watchlist from the newly built cache
        count = reload_watchlist()
        
        elapsed = time.time() - start
        print(f"[ENDPOINT] Full reload completed in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "message": "Cache rebuilt and watchlist reloaded successfully",
            "watchlist_size": count,
            "reload_time_seconds": round(elapsed, 2)
        }
        
    except Exception as e: 
        elapsed = time.time() - start
        print(f"[ERROR] Reload failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status":  "error",
            "message":  str(e),
            "reload_time_seconds": round(elapsed, 2)
        }

# ---------- Config endpoints ----------
@app.get("/config")
def get_config():
    return {
        "Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh,
        "w_hdic": w_hdic, "T_final": T_final,
        "admin_api": ADMIN_API
    }

@app.post("/config")
def update_config(payload: dict):
    global Tnh, Thdic, w_nh, w_hdic, T_final, ADMIN_API
    # Safely parse floats and update globals
    if "Tnh" in payload:
        try: Tnh = float(payload["Tnh"])
        except: pass
    if "Thdic" in payload:
        try: Thdic = float(payload["Thdic"])
        except: pass
    if "w_nh" in payload:
        try:
            v = float(payload["w_nh"])
            if 0.0 <= v <= 1.0: w_nh = v
        except: pass
    if "w_hdic" in payload:
        try:
            v = float(payload["w_hdic"])
            if 0.0 <= v <= 1.0: w_hdic = v
        except: pass
    if "T_final" in payload:
        try: T_final = float(payload["T_final"])
        except: pass
    if "admin_api" in payload:
        ADMIN_API = str(payload["admin_api"])

    cfg = {
        "Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": w_hdic,
        "T_final": T_final, "port": PORT,
        "log_path": str(LOGPATH.relative_to(Path(__file__).parent)),
        "admin_api": ADMIN_API
    }
    CFG_PATH.write_text(yaml.safe_dump(cfg))
    return {"ok": True, "config": cfg}

# ---------- Single-frame match ----------
@app.post("/match")
async def match(file: UploadFile = File(...)):
    global last_led_state

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

    decision = result.get("decision")

    # -------------------
    # LED Status Handling
    # -------------------
    if decision == "MATCH":
        target = "CRIMINAL"
    elif decision == "MANUAL_CHECK":
        target = "IDLE"  # or create a specific LED state for manual check
    elif decision == "NO_MATCH":
        target = "SAFE"
    elif decision == "NO_FACE": 
        target = "IDLE"
    else:
        target = "IDLE"

    # Avoid sending duplicate commands
    if target != last_led_state:
        led.send(target)
        last_led_state = target

    # Send to admin on match OR manual check
    if decision == "MANUAL_CHECK":
        pid = result.get("person_id") or "unknown"
        sfinal = (result.get("scores") or {}).get("Sfinal", 0.0)
        send_manual_alert(pid, sfinal, frame)

    return result

# ---------- Multi-frame match ----------
@app.post("/match_multi")
async def match_multi(files: List[UploadFile] = File(...)):
    frames_data = [(i, await uf.read(), uf.filename) for i, uf in enumerate(files)]

    def process_single(idx, img_bytes, filename):
        try:
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Frame decode failed → NO_FACE
            if frame is None:
                print(f"Frame {idx+1}: NO_FACE (decode error)")
                return {
                    "index": idx + 1,
                    "filename": filename,
                    "decision": "NO_FACE",
                    "ok": False,
                    "Sfinal": None,
                    "d_nh": None,
                    "d_hdic": None,
                    "person_id": None,
                }

            result = match_frame(
                frame_bgr=frame,
                Tnh=Tnh, Thdic=Thdic,
                w_nh=w_nh, w_hdic=w_hdic,
                fused_th=T_final,
            )

            # If match_frame explicitly reports no face
            if result.get("decision") == "NO_FACE":
                print(f"Frame {idx+1}: NO_FACE")
                return {
                    "index": idx + 1,
                    "filename": filename,
                    "decision": "NO_FACE",
                    "ok": False,
                    "Sfinal": None,
                    "d_nh": None,
                    "d_hdic": None,
                    "person_id": None,
                }

            # Extract computed values
            scores = result.get("scores", {}) or {}
            Sfinal = scores.get("Sfinal")
            d_nh = scores.get("d_nh")
            d_hdic = scores.get("d_hdic")
            pid = result.get("person_id")

            # Determine if match
            valid_face = (
                d_nh is not None and
                d_hdic is not None and
                Sfinal is not None
            )

            if not valid_face:
                print(f"Frame {idx+1}: NO_FACE or ERROR")
                return {
                    "index": idx + 1,
                    "filename": filename,
                    "decision": "NO_FACE",
                    "ok": False,
                    "Sfinal": None,
                    "d_nh": None,
                    "d_hdic": None,
                    "person_id": None,
                }

            # Determine match status with three-tier logic
            if (d_nh < Tnh and d_hdic < Thdic):
                if Sfinal >= 0.8:
                    ok = "MATCH"  # Strong match
                    frame_decision = "MATCH"
                elif 0.75 <= Sfinal < 0.8:
                    ok = "MANUAL"  # Needs manual verification
                    frame_decision = "MANUAL_CHECK"
                else:
                    ok = False
                    frame_decision = "NO_MATCH"
            else:
                ok = False
                frame_decision = "NO_MATCH"

            print(
                f"Frame {idx+1}: d_nh={int(d_nh)}, d_hdic={int(d_hdic)}, "
                f"Sfinal={float(Sfinal):.3f} -> {frame_decision}"
            )

            return {
                "index":  idx + 1,
                "filename": filename,
                "ok": ok,
                "decision":  frame_decision,
                "Sfinal": float(Sfinal),
                "d_nh": int(d_nh),
                "d_hdic":  int(d_hdic),
                "person_id": pid,
            }

        except Exception:
            print(f"Frame {idx+1}: ERROR → treated as NO_FACE")
            return {
                "index": idx + 1,
                "filename": filename,
                "decision": "NO_FACE",
                "ok": False,
                "Sfinal": None,
                "d_nh": None,
                "d_hdic": None,
                "person_id": None,
            }

    # Process frames in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda args: process_single(*args), frames_data))

    # Filter out NO_FACE frames when deciding MATCH/NO_MATCH
    valid_results = [r for r in results if r["decision"] != "NO_FACE"]

    # If no valid face detected in ANY frame → IDLE
    if not valid_results:
        led.send("IDLE")
        return {
            "decision": "NO_FACE",
            "frames": len(results),
            "match_frames": 0,
            "match_ratio": 0.0,
            "best_score": None,
            "best_person_id": None,
            "frame_details": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

    total_valid = len(valid_results)
    
    # Count frames in each category
    strong_match_count = sum(1 for r in valid_results if r["ok"] == "MATCH")  # Sfinal >= 0.8
    manual_check_count = sum(1 for r in valid_results if r["ok"] == "MANUAL")  # 0.75 <= Sfinal < 0.8
    no_match_count = sum(1 for r in valid_results if r["ok"] == False)  # Sfinal < 0.75

    # Decision logic based on frame counts
    if strong_match_count >= 3:
        # At least 3 frames with Sfinal >= 0.8 → Automatic MATCH
        majority_decision = "MATCH"

    elif strong_match_count == 2 and manual_check_count >= 1:
        # 2 frames with Sfinal >= 0.8 and at least 1 frame with 0.75 <= Sfinal < 0.8 → MANUAL_CHECK
        majority_decision = "MANUAL_CHECK"

    elif strong_match_count == 1 and manual_check_count >= 2:
        # 1 frame with Sfinal >= 0.8 and at least 2 frames with 0.75 <= Sfinal < 0.8 → MANUAL_CHECK
        majority_decision = "MANUAL_CHECK"
    
    elif manual_check_count >= 3:
        # At least 3 frames with 0.75 <= Sfinal < 0.8 → Manual verification needed
        majority_decision = "MANUAL_CHECK"
    else: 
        # Not enough strong evidence → NO_MATCH
        majority_decision = "NO_MATCH"

    # LED logic
    if majority_decision == "MATCH": 
        led.send("CRIMINAL")
    elif majority_decision == "MANUAL_CHECK":
        led.send("IDLE")  # or create a specific LED state
    else:
        led.send("SAFE")

    # Choose best scoring frame
    best_frame = max(valid_results, key=lambda x: x.get("Sfinal", 0))
    best_score = best_frame.get("Sfinal")
    best_pid = best_frame.get("person_id")
    best_idx = best_frame["index"] - 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Calculate match count for UI display
    match_count = strong_match_count + manual_check_count
    
    final_result = {
        "method": f"count-based-{total_valid}-valid",
        "frames": len(results),
        "valid_frames": total_valid,
        "match_frames": match_count,  # ✅ Total frames that matched (for UI)
        "match_ratio": round(match_count / total_valid, 2) if total_valid > 0 else 0.0,  # ✅ Match ratio
        "strong_match_frames": strong_match_count,  # Frames with Sfinal >= 0.8
        "manual_check_frames": manual_check_count,  # Frames with 0.75 <= Sfinal < 0.8
        "no_match_frames": no_match_count,  # Frames with Sfinal < 0.75
        "decision": majority_decision,
        "best_score":  round(best_score, 3),
        "best_person_id": best_pid if (majority_decision == "MATCH" or majority_decision == "MANUAL_CHECK") else None,
        "frame_details": results,
        "timestamp": timestamp,
    }

    log_match(final_result)

    # Trigger admin alert ONLY for MANUAL_CHECK
    # At least 3 frames with 0.75 <= Sfinal < 0.8 need manual verification
    if majority_decision == "MANUAL_CHECK" and best_pid:
        try:
            _, best_bytes, _ = frames_data[best_idx]
            enqueue_case(best_pid, best_score, best_bytes)
            send_manual_alert(best_pid, best_score, best_bytes, timestamp)
        except Exception:
            pass

    return final_result



# ✅ NEW: Admin feedback check endpoint
@app.get("/check_status")
def check_admin_status(person_id: str, timestamp: str):
    try:
        r = requests.get(
            f"{ADMIN_API}/manual_check/status",
            params={"person_id": person_id, "timestamp": timestamp},
            timeout=5,
            proxies={"http": None, "https": None}
        )
        return r.json()
    except Exception as e:
        print("[WARN] Failed to query admin for status:", e)
        return {"status": "unknown", "error": str(e)}


# ---------- Entry ----------
if __name__ == "__main__":
    uvicorn.run(
        "software_builds.field_client.backend.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=True,
    )
