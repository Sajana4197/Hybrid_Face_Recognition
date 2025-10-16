import os
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch

# ---------------- Face embedding ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes aligned RGB numpy image (160x160x3),
    returns a 512D embedding as float32 numpy array.
    """
    import torchvision.transforms as transforms

    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {rgb_image.shape}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize([0.5], [0.5])
    ])

    pil_img = Image.fromarray(rgb_image)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(img_tensor).cpu().numpy().flatten()

    return emb.astype(np.float32)


# ---------------- Neural Hash pipeline ---------------- #

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

    arr = np.fromfile(dat_path, dtype=np.int8).reshape(-1, 128).astype(np.float32)
    return arr[:96]   # ensure (96,128)

def compute_hash(embedding_128, hyperplanes):
    """
    embedding_128: (128,)
    hyperplanes: (96,128)
    Returns: 96-bit hash (numpy array of 0/1).
    """
    projections = np.dot(hyperplanes, embedding_128)  # (96,)
    return (projections > 0).astype(np.uint8)

def get_96d_neural_hash_from_array(rgb_image, pca, hyperplanes):
    """
    Full NeuralHash pipeline:
    - 512D embedding
    - PCA -> 128D
    - Hyperplane hashing -> 96-bit hash
    """
    emb512 = get_embedding(rgb_image)

    # Step 1: PCA (512 -> 128)
    emb128 = pca.transform([emb512])[0]

    # Step 2: Hyperplane hashing -> 96 bits
    bits96 = compute_hash(emb128, hyperplanes)

    return bits96
