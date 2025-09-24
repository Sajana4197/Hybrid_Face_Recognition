import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# Initialize GPU-accelerated MTCNN once
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=device)

def load_and_align(image_path, output_size=(160, 160), normalize=False):
    """
    Load an image, detect & align the largest face with MTCNN,
    and return an RGB uint8 face crop.
    If no valid face is found, returns None instead of raising.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is None or len(boxes) == 0:
        print(f"[WARN] No face detected in {image_path}")
        return None

    # Use largest detected face
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box

    # Clip coordinates to image boundaries
    h, w, _ = img_rgb.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        print(f"[WARN] Invalid face crop for {image_path}, box={box}")
        return None

    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        print(f"[WARN] Empty face crop for {image_path}, box={box}")
        return None

    # Resize to target size
    face = cv2.resize(face, output_size)

    if normalize:
        face = face.astype(np.float32) / 255.0
    else:
        face = face.astype(np.uint8)

    return face
