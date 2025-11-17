# neuralhash_dropout_sweep.py
import os, glob, pickle, csv
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1

# ===================== USER CONFIG =====================
TEST_FOLDER_PATH = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_hq"
PCA_PATH        = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\pca_512_to_128.pkl"
HYPERPLANES_PATH= r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\neuralhash_128x96_seed1.dat"

SWEEP_PS   = [0.2, 0.4, 0.6, 0.8]  # dropout probabilities to test
MC_SAMPLES = 50                          # forward passes per image for MC-Dropout
IMAGE_SIZE = 160
OUT_DIR    = "plots"
# =======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ----------------- Helpers -----------------
def load_pca(pca_path):
    with open(pca_path, "rb") as f:
        obj = pickle.load(f)
        return obj["pca_model"] if isinstance(obj, dict) and "pca_model" in obj else obj

def load_hyperplanes(dat_path):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"NeuralHash .dat file not found: {dat_path}")
    file_size = os.path.getsize(dat_path)
    dtype = np.float32
    expected_bytes = 128 * 96 * np.dtype(dtype).itemsize
    header_bytes = 32
    with open(dat_path, "rb") as f:
        if file_size > expected_bytes:
            f.seek(header_bytes)
            arr = np.fromfile(f, dtype=dtype, count=128 * 96)
        else:
            arr = np.fromfile(f, dtype=dtype)
    return arr.reshape(96, 128)

def compute_hash(embedding_512, pca, hyperplanes):
    emb128 = pca.transform([embedding_512])[0]
    n = np.linalg.norm(emb128)
    if n > 0:
        emb128 = emb128 / n
    projections = np.dot(hyperplanes, emb128)
    return (projections > 0).astype(np.uint8)

def bits_hamming(a, b):
    return int(np.sum(a != b))

def set_all_dropout_p(model, p):
    """Change p for every nn.Dropout present."""
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
            count += 1
    return count

def enable_only_dropout_training(model):
    """Keep BN/etc. frozen; only Dropout in train() so masks sample."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def list_dropout_layers(model):
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            layers.append((name, m.p))
    return layers

def detect_and_crop(mtcnn, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    face = mtcnn(img)
    if face is None:
        return None
    return face.unsqueeze(0).to(DEVICE)

# ----------------- Load models -----------------
print("Loading MTCNN + InceptionResnetV1 ...")
mtcnn  = MTCNN(image_size=IMAGE_SIZE, margin=0, keep_all=False, post_process=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").to(DEVICE).eval()

drops = list_dropout_layers(resnet)
if drops:
    print("Found Dropout layers:")
    for n, p in drops:
        print(f"  - {n}: p={p}")
else:
    print("⚠️ No nn.Dropout layers found in the model. MC-Dropout sweep will have no effect unless you insert Dropout.")

print("Loading PCA + NeuralHash hyperplanes ...")
pca_model  = load_pca(PCA_PATH)
hyperplanes= load_hyperplanes(HYPERPLANES_PATH)

# ----------------- Data -----------------
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, "**", ext), recursive=True))
if not image_paths:
    raise SystemExit(f"No images found in: {TEST_FOLDER_PATH}")
print(f"Found {len(image_paths)} images. Cropping faces ...")

faces, kept_paths = [], []
for p in image_paths:
    face = detect_and_crop(mtcnn, p)
    if face is not None:
        faces.append(face)
        kept_paths.append(p)
print(f"Usable faces: {len(faces)}/{len(image_paths)}")
if not faces:
    raise SystemExit("No faces detected; aborting.")

# ----------------- Sweep p values -----------------
os.makedirs(OUT_DIR, exist_ok=True)
mean_std_per_p = []      # y-axis for the bar chart
details_rows   = []      # per-image rows: relpath, p, std_hamming

for p in SWEEP_PS:
    print(f"\n=== Dropout p={p} ===")
    n_changed = set_all_dropout_p(resnet, p)
    if n_changed == 0:
        print("  (No Dropout layers to set)")

    # 1) Baseline stable hash (deterministic)
    resnet.eval()
    stable_bits_list = []
    with torch.no_grad():
        for face in faces:
            emb = resnet(face).cpu().numpy().flatten()
            stable_bits_list.append(compute_hash(emb, pca_model, hyperplanes))
    stable_bits = np.asarray(stable_bits_list, dtype=np.uint8)

    # 2) Enable only Dropout stochasticity
    enable_only_dropout_training(resnet)

    # 3) For each face, MC sample → STD of Hamming to baseline
    per_img_std = []
    with torch.no_grad():
        for idx, face in enumerate(faces):
            dists = []
            for _ in range(MC_SAMPLES):
                emb = resnet(face).cpu().numpy().flatten()
                nh = compute_hash(emb, pca_model, hyperplanes)
                dists.append(bits_hamming(stable_bits[idx], nh))
            dists = np.asarray(dists, dtype=np.float32)
            stdv  = float(dists.std(ddof=1)) if len(dists) > 1 else 0.0
            per_img_std.append(stdv)
            details_rows.append([
                os.path.relpath(kept_paths[idx], TEST_FOLDER_PATH), p, stdv
            ])

    mean_std = float(np.mean(per_img_std)) if per_img_std else 0.0
    mean_std_per_p.append(mean_std)
    print(f"Mean STD(Hamming) for p={p}: {mean_std:.4f}")

# ----------------- Plot: Dropout vs STD -----------------
x = np.arange(len(SWEEP_PS))
plt.figure()
plt.bar(x, mean_std_per_p)
plt.xticks(x, [str(v) for v in SWEEP_PS])
plt.xlabel("Dropout probability p")
plt.ylabel("Mean STD of Hamming distance")
plt.title(f"NeuralHash — Dropout vs Uncertainty (MC={MC_SAMPLES})")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "neuralhash_dropout_vs_std.png")
plt.savefig(out_png, dpi=150)
print(f"\nSaved plot → {out_png}")

# ----------------- Save details CSV -----------------
csv_path = os.path.join(OUT_DIR, "neuralhash_dropout_vs_std_details.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_relpath", "dropout_p", "std_hamming"])
    w.writerows(details_rows)
print(f"Saved details → {csv_path}")
