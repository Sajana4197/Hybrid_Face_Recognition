import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from keras_facenet import FaceNet
from facenet_pytorch import MTCNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt   # <-- added for plotting

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
base_model = embedder.model   # keep original model as base

# 3. Load Hypervector Projection Matrix
print("Creating Hypervector Projection Matrix...")
DIM_ORIG = 512
DIM_HV = 10000
np.random.seed(42)  # Ensure the matrix is consistent
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


def build_model_with_dropout(model, new_rate):
    """
    Clone the FaceNet model but replace its (single) Dropout layer
    with a new Dropout using new_rate.
    """
    dropout_index = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dropout):
            dropout_index = i
            break

    if dropout_index is None:
        raise ValueError("No Dropout layer found in FaceNet model.")

    print(f"  Found Dropout layer at index {dropout_index} with rate={model.layers[dropout_index].rate}")
    print(f"  Replacing it with new dropout rate = {new_rate}")

    # Rebuild graph from just before dropout onwards
    x = model.layers[dropout_index - 1].output
    x = Dropout(new_rate, name="custom_dropout")(x)

    for layer in model.layers[dropout_index + 1:]:
        x = layer(x)

    new_model = Model(inputs=model.input, outputs=x)
    print("  ✅ New model with custom dropout created.")
    return new_model


# ----------------- Main UQ Function (modified to take model) -----------------

def get_hv_and_uncertainty_from_path(image_path, mc_model, num_samples=50):
    """
    Main function to run the full UQ pipeline on a single image path,
    using the provided mc_model (FaceNet with chosen dropout rate).

    Returns:
        stable_hv,
        mean_hamming_distance,
        std_of_hamming_distance   <-- std, not variance
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
    # Uses original embedder (base_model internally) with dropout OFF (training=False)
    stable_emb512 = embedder.embeddings([face_numpy_uint8])[0]
    stable_hv = encode_embedding_to_hv(stable_emb512)

    # --- 3. Activate MC Dropout Mode in Keras Model (mc_model) ---
    for layer in mc_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # --- 4. Run N samples to get dropout HVs ---
    distance_list = []

    face_numpy_float = face_numpy_uint8.astype(np.float32)
    input_processed = (face_numpy_float - 127.5) / 127.5
    input_batch = np.expand_dims(input_processed, axis=0)

    for i in range(num_samples):
        dropout_emb512 = mc_model(input_batch, training=True)[0].numpy()
        dropout_hv = encode_embedding_to_hv(dropout_emb512)
        distance = calculate_hamming_distance(stable_hv, dropout_hv)
        distance_list.append(distance)

    # --- 5. Clean up ---
    for layer in mc_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # --- 6. Calculate Stats (MEAN + STD, not variance) ---
    distances_array = np.array(distance_list)
    mean_hamming_distance = np.mean(distances_array)
    std_of_hamming_distance = np.std(distances_array)

    return stable_hv, mean_hamming_distance, std_of_hamming_distance


# ----------------- Main Execution Block (MULTI-DROPOUT TEST) -----------------

if __name__ == "__main__":

    # --- Folder with test images ---
    TEST_FOLDER_PATH = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_hq"

    # Find all images in the folder and its subfolders
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '**', ext), recursive=True))

    if not image_paths:
        print(f"No images found in: {TEST_FOLDER_PATH}")
        exit()

    print(f"Found {len(image_paths)} images to process.")

    # --- Dropout rates to test ---
    DROPOUT_RATES = [0.2, 0.4, 0.6, 0.8]   # change this list as you like

    # Store overall results per dropout rate: (dr, avg_mean, avg_std)
    summary_results = []

    for dr in DROPOUT_RATES:
        print("\n" + "=" * 60)
        print(f"Testing dropout rate = {dr}")
        print("=" * 60)

        # Build a FaceNet model with this dropout rate
        mc_model = build_model_with_dropout(base_model, new_rate=dr)

        all_mean_distances = []
        all_stds = []
        processed_count = 0

        # Loop through all images for this dropout rate
        for image_path in image_paths:
            print(f"\n--- Processing: {os.path.basename(image_path)} ---")

            results = get_hv_and_uncertainty_from_path(
                image_path,
                mc_model=mc_model,
                num_samples=50  # Number of dropout runs per image
            )

            if results:
                stable_hv, mean_dist, std_dist = results
                all_mean_distances.append(mean_dist)
                all_stds.append(std_dist)
                processed_count += 1
                print(f"  Mean Dist: {mean_dist:.4f}, Std: {std_dist:.4f}")

        if processed_count > 0:
            avg_mean_dist = np.mean(all_mean_distances)
            avg_std_dist = np.mean(all_stds)

            print("\n" + "-" * 50)
            print(f"Dropout {dr} SUMMARY")
            print("-" * 50)
            print(f"  Total Images Processed: {processed_count} / {len(image_paths)}")
            print(f"  Average Mean Hamming Dist: {avg_mean_dist:.4f}")
            print(f"  Average Std of Dist:       {avg_std_dist:.4f}")
            print("-" * 50)

            summary_results.append((dr, avg_mean_dist, avg_std_dist))
        else:
            print(f"\nNo images were processed successfully for dropout={dr}.")

    # ---- Final overview over all tested dropout rates ----
    if summary_results:
        print("\n" + "=" * 60)
        print("      OVERALL COMPARISON OF DROPOUT RATES")
        print("=" * 60)
        for dr, avg_md, avg_std in summary_results:
            print(f"Dropout {dr}:  Avg Mean Dist = {avg_md:.4f},  Avg Std = {avg_std:.4f}")
        print("=" * 60)

        # ---------- BAR CHART: Dropout vs Std Deviation ----------
        dropout_vals = [r[0] for r in summary_results]
        avg_std_vals = [r[2] for r in summary_results]

        plt.figure()
        plt.bar([str(d) for d in dropout_vals], avg_std_vals)
        plt.xlabel("Dropout rate")
        plt.ylabel("Average standard deviation of Hamming distance")
        plt.title("MC-Dropout Uncertainty vs Dropout Rate")
        plt.tight_layout()
        # good for paper: high-res PNG
        plt.savefig("uq_dropout_std_barchart.png", dpi=300)
        plt.show()

    else:
        print("\nNo results to summarize across dropout rates.")
