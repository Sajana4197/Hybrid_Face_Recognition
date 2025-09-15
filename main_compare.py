import os
import pickle
import numpy as np
import joblib
from tqdm import tqdm
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# ---------------- Face detection & embedding ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_face(img_path):
    """Detect and crop a face, return as tensor or None if no face found."""
    img = Image.open(img_path).convert("RGB")
    face_tensor = mtcnn(img)
    return face_tensor

def get_embedding(face_tensor):
    """Convert cropped face to 512-D embedding."""
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().flatten()
    return emb

# ---------------- Neural Hash pipeline ---------------- #

def load_pca(pca_path):
    try:
        with open(pca_path, 'rb') as f:
            pca_data = pickle.load(f)
            if isinstance(pca_data, dict) and 'pca_model' in pca_data:
                return pca_data['pca_model']
            else:
                # If it's directly a PCA model
                return pca_data
    except Exception as e:
        raise ValueError(f"Error loading PCA model: {e}")

# Load projection matrix and hyperplanes from .dat
def load_projection_and_hyperplanes(dat_path):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Data file not found: {dat_path}")
    
    with open(dat_path, 'rb') as f:
        raw = f.read()
    
    print(f"File size: {len(raw)} bytes")
    
    header_size = 128
    matrix_size = 128 * 96 * 4  # float32 = 4 bytes per element
    expected_total_size = header_size + 2 * matrix_size  # header + proj_matrix + hyperplanes
    
    print(f"Expected file size: {expected_total_size} bytes")
    print(f"Matrix size: {matrix_size} bytes each")
    
    if len(raw) < header_size + matrix_size:
        raise ValueError(f"File too small. Expected at least {header_size + matrix_size} bytes, got {len(raw)}")
    
    # Extract projection matrix
    proj_start = header_size
    proj_end = header_size + matrix_size
    proj_flat = np.frombuffer(raw[proj_start:proj_end], dtype=np.float32)
    
    if len(proj_flat) != 128 * 96:
        raise ValueError(f"Projection matrix has wrong size: {len(proj_flat)}, expected {128 * 96}")
    
    proj_matrix = proj_flat.reshape(128, 96)
    
    # Check if hyperplanes data exists
    if len(raw) >= header_size + 2 * matrix_size:
        # Extract hyperplanes
        hyper_start = header_size + matrix_size
        hyper_end = header_size + 2 * matrix_size
        hyper_flat = np.frombuffer(raw[hyper_start:hyper_end], dtype=np.float32)
        
        if len(hyper_flat) != 128 * 96:
            print(f"Warning: Hyperplanes data has size {len(hyper_flat)}, expected {128 * 96}")
            print("Generating random hyperplanes as fallback...")
            hyperplanes = generate_random_hyperplanes()
        else:
            hyperplanes = hyper_flat.reshape(128, 96)
    else:
        print("Warning: File doesn't contain hyperplanes data. Generating random hyperplanes...")
        hyperplanes = generate_random_hyperplanes()
    
    return proj_matrix, hyperplanes

# Generate random hyperplanes if not available in file
def generate_random_hyperplanes():
    """Generate random hyperplanes for hashing when not available in data file"""
    np.random.seed(42)  # For reproducible results
    hyperplanes = np.random.randn(128, 96)
    # Normalize each hyperplane
    for i in range(128):
        hyperplanes[i] = hyperplanes[i] / np.linalg.norm(hyperplanes[i])
    return hyperplanes


def compute_hash(embedding_96, hyperplanes):
    return np.dot(hyperplanes, embedding_96) > 0

def bits_to_hex(bits):
    byte_array = np.packbits(bits.astype(np.uint8))
    return "".join(f"{b:02x}" for b in byte_array)

def get_96d_neural_hash(img_path, pca, proj_matrix):
    """Full pipeline: face → 512D emb → PCA (128D) → project (96D)."""
    face_tensor = load_face(img_path)
    if face_tensor is None:
        return None
    emb512 = get_embedding(face_tensor)
    if emb512 is None:
        return None
    emb128 = pca.transform([emb512])[0]
    emb96 = np.dot(emb128, proj_matrix)
    return emb96

# ---------------- Database builder ---------------- #

def build_criminal_db(dataset_root,
                      pca,
                      proj_matrix,
                      hyperplanes,
                      save_path="./criminal_neuralhash_db.pkl"):
    db = {}
    person_dirs = [d for d in os.listdir(dataset_root)
                   if os.path.isdir(os.path.join(dataset_root, d))]

    # outer progress bar over persons
    for person in tqdm(person_dirs, desc="Building DB (persons)", unit="person"):
        person_path = os.path.join(dataset_root, person)
        images = [os.path.join(person_path, f)
                  for f in os.listdir(person_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if not images:
            tqdm.write(f"[WARN] No images found for {person}, skipping.")
            continue

        saved_hashes, saved_hex, saved_embs96, saved_images = [], [], [], []

        # inner progress bar over images for this person; leave=False so it clears after person done
        for img_path in tqdm(images, desc=f"  Images for {person}", leave=False, unit="img"):
            try:
                emb96 = get_96d_neural_hash(img_path, pca, proj_matrix)
                if emb96 is None:
                    continue
                emb96_norm = emb96 / np.linalg.norm(emb96)
                bits = compute_hash(emb96_norm, hyperplanes)
                hexstr = bits_to_hex(bits)

                saved_hashes.append(bits)
                saved_hex.append(hexstr)
                saved_embs96.append(emb96_norm)
                saved_images.append(img_path)
            except Exception as e:
                tqdm.write(f"[WARN] {person} -> {img_path}: {e}")
                continue

        if not saved_embs96:
            tqdm.write(f"[WARN] No valid faces for {person}, skipping.")
            continue

        mean_emb = np.mean(np.stack(saved_embs96), axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        mean_bits = compute_hash(mean_emb, hyperplanes)

        db[person] = {
            "images": saved_images,
            "hash_bits": saved_hashes,
            "hash_hex": saved_hex,
            "embeddings_96": saved_embs96,
            "mean_embedding_96": mean_emb,
            "mean_hash_bits": mean_bits,
            "mean_hash_hex": bits_to_hex(mean_bits),
        }

        # quick per-person summary (printed via tqdm.write so it doesn't break bars)
        tqdm.write(f"[INFO] {person}: processed={len(images)}, valid_faces={len(saved_embs96)}")

    with open(save_path, "wb") as f:
        pickle.dump(db, f)
    tqdm.write(f"✅ DB saved to {save_path} (entries: {len(db)})")
    return db

# ---------------- Matching ---------------- #

def match_probe_to_db(probe_img_path,
                      pca,
                      proj_matrix,
                      hyperplanes,
                      db,
                      top_k=5,
                      ham_threshold=12,
                      cos_threshold=0.65):
    emb96 = get_96d_neural_hash(probe_img_path, pca, proj_matrix)
    if emb96 is None:
        raise ValueError("Could not extract embedding for probe.")
    emb96 = emb96 / np.linalg.norm(emb96)
    probe_bits = compute_hash(emb96, hyperplanes)

    results = []
    for person, rec in db.items():
        hams = [int(np.sum(probe_bits != h)) for h in rec["hash_bits"]]
        min_ham = min(hams)
        mean_ham = int(np.sum(probe_bits != rec["mean_hash_bits"]))
        cos = float(np.dot(emb96, rec["mean_embedding_96"]))

        # Decision rule
        is_match = (min_ham <= ham_threshold) and (cos >= cos_threshold)

        results.append({
            "person": person,
            "min_hamming": min_ham,
            "mean_hamming": mean_ham,
            "cosine": cos,
            "example_image": rec["images"][0] if rec["images"] else None,
            "match_decision": is_match
        })

    results.sort(key=lambda r: (r["min_hamming"], -r["cosine"]))
    return results[:top_k]

# ---------------- CLI ---------------- #

def load_db(db_path):
    with open(db_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Criminal Neural Hash DB")
    parser.add_argument("--mode", choices=["build", "match"], required=True)
    parser.add_argument("--dataset_root", help="Root folder of criminals dataset")
    parser.add_argument("--db_path", default="./criminal_neuralhash_db.pkl")
    parser.add_argument("--probe", help="Probe image path")
    parser.add_argument("--pca_file", default="./models/pca_512_to_128.pkl")
    parser.add_argument("--dat_file", default="./models/neuralhash_128x96_seed1.dat")
    parser.add_argument("--ham_thresh", type=int, default=12)
    parser.add_argument("--cos_thresh", type=float, default=0.65)
    args = parser.parse_args()

    pca_model = load_pca(args.pca_file)
    proj_matrix, hyperplanes = load_projection_and_hyperplanes(args.dat_file)

    if args.mode == "build":
        build_criminal_db(args.dataset_root, pca_model, proj_matrix, hyperplanes,
                          save_path=args.db_path)

    elif args.mode == "match":
        db = load_db(args.db_path)
        results = match_probe_to_db(args.probe, pca_model, proj_matrix,
                                    hyperplanes, db,
                                    top_k=5,
                                    ham_threshold=args.ham_thresh,
                                    cos_threshold=args.cos_thresh)
        print("\nTop matches:")
        for r in results:
            decision = "✅ MATCH" if r["match_decision"] else "❌ NO MATCH"
            print(f"Person: {r['person']}, min_hamming={r['min_hamming']}, "
                  f"cosine={r['cosine']:.3f}, decision={decision}, "
                  f"example_image={r['example_image']}")
