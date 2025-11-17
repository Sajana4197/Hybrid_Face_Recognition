import os
import glob # Added for finding image files
import numpy as np
from PIL import Image
import tensorflow as tf
from keras_facenet import FaceNet
from facenet_pytorch import MTCNN

# ----------------- Configuration & Model Loading -----------------

# 1. Load Face Detector (MTCNN)
print("Loading Face Detector (MTCNN)...")
detector = MTCNN(
    image_size=160, 
    margin=0, 
    keep_all=False, 
    post_process=True
)

# 2. Load Face Embedder (Keras FaceNet)
print("Loading Face Embedder (Keras FaceNet)...")
embedder = FaceNet()
keras_model = embedder.model

# 3. Load Hypervector Projection Matrix
print("Creating Hypervector Projection Matrix...")
DIM_ORIG = 512
DIM_HV = 10000
np.random.seed(42) # Ensure the matrix is consistent
projection_matrix = np.random.randn(DIM_ORIG, DIM_HV)


# ----------------- Helper Functions -----------------

def encode_embedding_to_hv(embedding):
    """
    Converts a 512D embedding to a 10,000D binary hypervector.
    """
    projected = np.dot(embedding, projection_matrix)
    hypervector = (projected > 0).astype(np.uint8)
    return hypervector

def calculate_hamming_distance(h1, h2):
    """Calculates the Hamming distance between two numpy bit arrays."""
    return np.sum(h1 != h2)


# ----------------- Main UQ Function (Unchanged) -----------------

def get_hv_and_uncertainty_from_path(image_path, num_samples=50):
    """
    Main function to run the full UQ pipeline on a single image path.
    """
    
    # --- 1. Detect and Crop Face ---
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
    face_tensor = detector(img) 
    
    if face_tensor is None:
        print(f"  Skipping (No face detected).")
        return None

    face_numpy_uint8 = face_tensor.permute(1, 2, 0).numpy()
    face_numpy_uint8 = ((face_numpy_uint8 + 1) * 127.5).astype(np.uint8)

    # --- 2. Get Stable "Ground Truth" HV (Eval Mode) ---
    stable_emb512 = embedder.embeddings([face_numpy_uint8])[0]
    stable_hv = encode_embedding_to_hv(stable_emb512)

    # --- 3. Activate MC Dropout Mode in Keras Model ---
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # --- 4. Run N samples to get dropout HVs ---
    distance_list = []
    
    face_numpy_float = face_numpy_uint8.astype(np.float32)
    input_processed = (face_numpy_float - 127.5) / 127.5
    input_batch = np.expand_dims(input_processed, axis=0)

    for i in range(num_samples):
        dropout_emb512 = keras_model(input_batch, training=True)[0].numpy()
        dropout_hv = encode_embedding_to_hv(dropout_emb512)
        distance = calculate_hamming_distance(stable_hv, dropout_hv)
        distance_list.append(distance)

    # --- 5. Clean up ---
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # --- 6. Calculate Stats ---
    distances_array = np.array(distance_list)
    mean_hamming_distance = np.mean(distances_array)
    variance_of_hamming_distance = np.var(distances_array)

    return stable_hv, mean_hamming_distance, variance_of_hamming_distance


# ----------------- Main Execution Block (MODIFIED) -----------------

if __name__ == "__main__":
    
    # --- ⚠️ CHANGE THIS PATH ---
    # This should be the folder containing all your test images
    TEST_FOLDER_PATH = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_ijbb"
    # ----------------------------

    # Find all images in the folder and its subfolders
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        # Use recursive=True to find images in subdirectories
        image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '**', ext), recursive=True))

    if not image_paths:
        print(f"No images found in: {TEST_FOLDER_PATH}")
        exit()
        
    print(f"Found {len(image_paths)} images to process.")

    # Lists to store results from all images
    all_mean_distances = []
    all_variances = []
    processed_count = 0

    # Loop through all found images
    for image_path in image_paths:
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")
        
        results = get_hv_and_uncertainty_from_path(
            image_path, 
            num_samples=50 # Number of dropout runs per image
        )

        if results:
            stable_hv, mean_dist, var_dist = results
            
            # Store the results for final calculation
            all_mean_distances.append(mean_dist)
            all_variances.append(var_dist)
            processed_count += 1
            
            # Print per-image results
            print(f"  Mean Dist: {mean_dist:.4f}, Variance: {var_dist:.4f}")
    
    # --- Final Summary Report ---
    if processed_count > 0:
        # Calculate the average of all results
        avg_mean_dist = np.mean(all_mean_distances)
        avg_variance = np.mean(all_variances)

        print("\n" + "=" * 50)
        print("         OVERALL HDIC UNCERTAINTY SUMMARY")
        print("=" * 50)
        print(f"Total Images Processed: {processed_count} / {len(image_paths)}")
        print(f"Average Mean Hamming Dist: {avg_mean_dist:.4f}")
        print(f"Average Variance of Dist:  {avg_variance:.4f}  <-- OVERALL UQ SCORE")
        print("-" * 50)
        
        # Example interpretation of the *average* score
        if avg_variance > 5000.0: # EXAMPLE THRESHOLD
            print("Status: ⚠️ High average uncertainty. The model is generally unstable.")
        else:
            print("Status: ✅ Low average uncertainty. The model is generally stable.")
    else:
        print("\nNo images were processed successfully.")