import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import glob # Added for folder scanning

# ----------------- Configuration & Model Loading -----------------

# Set up device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
# 1. MTCNN for face detection and cropping
print("Loading Face Detector (MTCNN)...")
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    keep_all=False, 
    post_process=True, 
    device=device
)

# 2. InceptionResnetV1 for embedding generation
print("Loading Face Embedder (InceptionResnetV1)...")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ----------------- Helper Functions -----------------

def load_pca(pca_path):
    """Load PCA model (512 -> 128)."""
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
        if isinstance(pca_data, dict) and "pca_model" in pca_data:
            return pca_data["pca_model"]
        return pca_data

def load_hyperplanes(dat_path):
    """
    Load Apple's NeuralHash hyperplanes (96x128) from .dat file.
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"NeuralHash .dat file not found: {dat_path}")
    
    file_size = os.path.getsize(dat_path)
    dtype = np.float32
    bytes_per_elem = np.dtype(dtype).itemsize
    expected_bytes = 128 * 96 * bytes_per_elem
    header_bytes = 32

    with open(dat_path, "rb") as f:
        # Handle files with a header
        if file_size > expected_bytes:
            f.seek(header_bytes)
            arr = np.fromfile(f, dtype=dtype, count=128 * 96)
        else:
            arr = np.fromfile(f, dtype=dtype)
    
    return arr.reshape(96, 128)


def compute_hash(embedding_512, pca, hyperplanes):
    """
    Converts a 512D embedding into a 96-bit hash.
    """
    emb128 = pca.transform([embedding_512])[0]
    emb128 /= np.linalg.norm(emb128)
    projections = np.dot(hyperplanes, emb128)
    return (projections > 0).astype(np.uint8)

def bits_to_hex(bits):
    """Converts a numpy array of bits to a hex string."""
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0,len(bits),8))

def calculate_hamming_distance(bits1, bits2):
    """Calculates the Hamming distance between two numpy bit arrays."""
    return np.sum(bits1 != bits2)


# ----------------- Main UQ Function (Modified for quiet batch processing) -----------------

def get_hash_and_uncertainty_from_path(image_path, pca_model, hyperplanes_data, num_samples=50):
    """
    Main function to run the full UQ pipeline on a single image path.
    """
    
    # --- 1. Detect and Crop Face ---
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"  Skipping (File not found).")
        return None # Return None on failure
    
    face_tensor = mtcnn(img) 
    
    if face_tensor is None:
        print(f"  Skipping (No face detected).")
        return None # Return None on failure

    face_tensor = face_tensor.unsqueeze(0).to(device)

    # --- 2. Get Stable "Ground Truth" Hash ---
    resnet.eval()
    with torch.no_grad():
        stable_emb512 = resnet(face_tensor).cpu().numpy().flatten()
    
    stable_hash_bits = compute_hash(stable_emb512, pca_model, hyperplanes_data)

    # --- 3. Activate MC Dropout Mode ---
    resnet.eval()
    for module in resnet.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

    # --- 4. Run N samples to get dropout hashes ---
    distance_list = []
    
    with torch.no_grad():
        for i in range(num_samples):
            dropout_emb512 = resnet(face_tensor).cpu().numpy().flatten()
            dropout_hash_bits = compute_hash(dropout_emb512, pca_model, hyperplanes_data)
            distance = calculate_hamming_distance(stable_hash_bits, dropout_hash_bits)
            distance_list.append(distance)

    # --- 5. Clean up ---
    resnet.eval() 

    # --- 6. Calculate Stats ---
    distances_array = np.array(distance_list)
    mean_hamming_distance = np.mean(distances_array)
    variance_of_hamming_distance = np.var(distances_array)

    # Return results instead of None
    return stable_hash_bits, mean_hamming_distance, variance_of_hamming_distance


# ----------------- Main Execution Block (MODIFIED FOR BATCH PROCESSING) -----------------

if __name__ == "__main__":
    
    # --- ⚠️ CHANGE THESE THREE PATHS ---
    TEST_FOLDER_PATH = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_hq" # FOLDER of images
    PCA_PATH = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\pca_512_to_128.pkl"
    HYPERPLANES_PATH = r"D:\FYP\Hybrid_Face_Recognition\neuralhash\assets\neuralhash_128x96_seed1.dat"
    # ------------------------------------

    print("Loading PCA and Hyperplane models...")
    try:
        pca_model = load_pca(PCA_PATH)
        hyperplanes_data = load_hyperplanes(HYPERPLANES_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()

    # --- Find all images in the folder ---
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        # Use recursive=True to find images in subdirectories
        image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '**', ext), recursive=True))

    if not image_paths:
        print(f"No images found in: {TEST_FOLDER_PATH}")
        exit()
        
    print(f"Found {len(image_paths)} images to process.")

    # --- Lists to store results from all images ---
    all_mean_distances = []
    all_variances = []
    processed_count = 0

    # --- Loop through all found images ---
    for image_path in image_paths:
        print(f"\n--- Processing: {os.path.relpath(image_path, TEST_FOLDER_PATH)} ---")
        
        results = get_hash_and_uncertainty_from_path(
            image_path, 
            pca_model, 
            hyperplanes_data,
            num_samples=50 # Number of dropout runs per image
        )

        if results:
            stable_bits, mean_dist, var_dist = results
            
            # Store the results for final calculation
            all_mean_distances.append(mean_dist)
            all_variances.append(var_dist)
            processed_count += 1
            
            # Print per-image results
            print(f"  Mean Dist: {mean_dist:.4f}, Variance (UQ Score): {var_dist:.4f}")
    
    # --- Final Summary Report ---
    if processed_count > 0:
        # Calculate the average of all results
        avg_mean_dist = np.mean(all_mean_distances)
        avg_variance = np.mean(all_variances)

        print("\n" + "=" * 50)
        print("         OVERALL UNCERTAINTY SUMMARY (NeuralHash)")
        print("=" * 50)
        print(f"Total Images Processed: {processed_count} / {len(image_paths)}")
        print(f"Average Mean Hamming Dist: {avg_mean_dist:.4f}")
        print(f"Average Variance of Dist:  {avg_variance:.4f}  <-- OVERALL UQ SCORE")
        print("-" * 50)
        
        # Example interpretation of the *average* score
        if avg_variance > 10.0: # Your previous threshold
            print("Status: ⚠️ High average uncertainty. The model is generally unstable.")
        else:
            print("Status: ✅ Low average uncertainty. The model is generally stable.")
    else:
        print("\nNo images were processed successfully.")