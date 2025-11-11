import numpy as np, cv2, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from software_builds.common_core.cache import matcher
from software_builds.common_core.settings import T_NH, T_HD, W_NH, W_HD, FUSED_TH

def match_ndarray(img_bgr: np.ndarray):
    return matcher().match_frame(img_bgr, Tnh=T_NH, Thdic=T_HD, w_nh=W_NH, w_hdic=W_HD, fused_th=FUSED_TH)

def match_bytes(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")
    return match_ndarray(img)
