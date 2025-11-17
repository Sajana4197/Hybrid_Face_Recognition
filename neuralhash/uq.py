import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import glob

# ----------------- Configuration & Model Loading -----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Face Detector (MTCNN)...")
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    keep_all=False, 
    post_process=True, 
    device=device
)

print("Loading Face Embedder (InceptionResnetV1)...")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ---- Force all dropout layers to p=0.2 ----
changed = 0
for m in resnet.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.2
        changed += 1
print(f"✅ Set dropout probability = 0.2 for {changed} dropout layer(s)")

# ----------------- Helper Functions -----------------
def load_pca(pca_path):
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
        if isinstance(pca_data, dict) and "pca_model" in pca_data:
            return pca_data["pca_model"]
        return pca_data

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
    emb128 /= np.linalg.norm(emb128)
    projections = np.dot(hyperplanes, emb128)
    return (projections > 0).astype(np.uint8)

def calculate_hamming_distance(bits1, bits2):
    return np.sum(bits1 != bits2)

# ----------------- Main Function -----------------
def get_hash_and_uncertainty_from_path(image_path, pca_model, hyperplanes_data, num_samples=50):
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"  Skipping (File not found).")
        return None
    
    face_tensor = mtcnn(img)
    if face_tensor is None:
        print(f"  Skipping (No face detected).")
        return None

    face_tensor = face_tensor.unsqueeze(0).to(device)

    # --- Stable embedding (baseline) ---
    resnet.eval()
    with torch.no_grad():
        stable_emb512 = resnet(face_tensor).cpu().numpy().flatten()
    stable_hash_bits = compute_hash(stable_emb512, pca_model, hyperplanes_data)

    # --- Enable MC Dropout ---
    resnet.eval()
    for module in resnet.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    distance_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            dropout_emb512 = resnet(face_tensor).cpu().numpy().flatten()
            dropout_hash_bits = compute_hash(dropout_emb512, pca_model, hyperplanes_data)
            d = calculate_hamming_distance(stable_hash_bits, dropout_hash_bits)
            distance_list.append(d)

    resnet.eval()

    distances_array = np.array(distance_list)
    mean_hamming_distance = np.mean(distances_array)
    variance_of_hamming_distance = np.var(distances_array)
    return stable_hash_bits, mean_hamming_distance, variance_of_hamming_distance

# ----------------- Execution -----------------
if __name__ == "__main__":
    TEST_FOLDER_PATH = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_ijbb"
    PCA_PATH = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\pca_512_to_128.pkl"
    HYPERPLANES_PATH = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\neuralhash_128x96_seed1.dat"

    print("Loading PCA and Hyperplane models...")
    try:
        pca_model = load_pca(PCA_PATH)
        hyperplanes_data = load_hyperplanes(HYPERPLANES_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()

    # Gather image paths
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '**', ext), recursive=True))
    if not image_paths:
        print(f"No images found in: {TEST_FOLDER_PATH}")
        exit()

    print(f"Found {len(image_paths)} images to process.")
    all_mean_distances, all_variances = [], []
    processed_count = 0

    for image_path in image_paths:
        print(f"\n--- Processing: {os.path.relpath(image_path, TEST_FOLDER_PATH)} ---")
        results = get_hash_and_uncertainty_from_path(image_path, pca_model, hyperplanes_data, num_samples=50)
        if results:
            _, mean_dist, var_dist = results
            all_mean_distances.append(mean_dist)
            all_variances.append(var_dist)
            processed_count += 1
            print(f"  Mean Dist: {mean_dist:.4f}, Variance (UQ): {var_dist:.4f}")

    if processed_count > 0:
        avg_mean = np.mean(all_mean_distances)
        avg_var  = np.mean(all_variances)
        print("\n" + "=" * 50)
        print("      OVERALL UNCERTAINTY SUMMARY (NeuralHash, Dropout=0.2)")
        print("=" * 50)
        print(f"Images Processed: {processed_count} / {len(image_paths)}")
        print(f"Avg Mean Hamming Dist: {avg_mean:.4f}")
        print(f"Avg Variance of Dist:  {avg_var:.4f}  <-- OVERALL UQ SCORE")
        print("-" * 50)
        if avg_var > 10.0:
            print("Status: ⚠️ High uncertainty (unstable embeddings).")
        else:
            print("Status: ✅ Stable embeddings (low uncertainty).")
    else:
        print("\nNo images processed successfully.")
