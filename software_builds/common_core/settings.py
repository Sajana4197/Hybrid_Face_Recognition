from pathlib import Path
import os

# HYBRID_FACE_RECOGNITION/
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_DIR    = REPO_ROOT / "db"
NH_DIR    = DB_DIR / "nh_packed"
HD_DIR    = DB_DIR / "hdic_packed"

# Tunables (can also be overridden by env vars)
T_NH     = float(os.getenv("T_NH", 30))
T_HD     = float(os.getenv("T_HD", 3100))
W_NH     = float(os.getenv("W_NH", 0.4))
W_HD     = float(os.getenv("W_HD", 0.6))
FUSED_TH = float(os.getenv("FUSED_TH", 0.75))
