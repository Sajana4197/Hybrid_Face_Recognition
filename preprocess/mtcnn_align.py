# MTCNN_preprocess_gpu.py

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

# === Parameters ===
DATASET_ROOT = 'images/raw'
OUTPUT_ROOT = 'images/cropped'
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
NUM_WORKERS = 8
BLUR_THRESHOLD = 40.0
TARGET_SIZE = (160, 160)
PADDING = 10

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging to write into logs/face_crop.log
logging.basicConfig(
    filename=os.path.join("logs", "face_crop.log"),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# === Initialize GPU-accelerated MTCNN ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS

def align_and_crop_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logging.warning(f"⚠️ Failed to read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces with landmarks
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if boxes is None or len(boxes) == 0:
        logging.info(f"No face detected in {image_path}")
        return None

    # Select largest face by box area
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    box = boxes[idx].astype(int)
    landmark = landmarks[idx]

    try:
        # Eye coordinates for alignment
        left_eye, right_eye = landmark[0], landmark[1]
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

        # Rotate image around eye center to align eyes horizontally
        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(img_rgb, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        # Adjust bounding box coords for padding and clamp to image size
        x1 = max(0, box[0] - PADDING)
        y1 = max(0, box[1] - PADDING)
        x2 = min(aligned.shape[1], box[2] + PADDING)
        y2 = min(aligned.shape[0], box[3] + PADDING)

        cropped = aligned[y1:y2, x1:x2]
        if cropped.size == 0:
            logging.info(f"Empty crop after alignment: {image_path}")
            return None

        resized = cv2.resize(cropped, TARGET_SIZE)

        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < BLUR_THRESHOLD:
            logging.info(f"Skipped blurry image (var={lap_var:.2f}): {image_path}")
            return None

        return resized

    except Exception as e:
        logging.warning(f"Alignment failed for {image_path}: {e}")
        return None
    
def align_and_crop_face_from_array(
    img_bgr: np.ndarray,
    mtcnn: MTCNN,
    padding: int = 10,
    target_size: tuple[int, int] = (160, 160),
    blur_threshold: float = 40.0
) -> np.ndarray | None:

    if img_bgr is None or img_bgr.size == 0:
        logging.warning("Empty input image")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if boxes is None or len(boxes) == 0:
        return None

    # Select largest face by bounding box area
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    box = boxes[idx].astype(int)
    landmark = landmarks[idx]

    try:
        # Eye coordinates for alignment
        left_eye, right_eye = landmark[0], landmark[1]
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

        # Rotate image around eye center to align eyes horizontally
        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(img_rgb, rot_mat, (img_bgr.shape[1], img_bgr.shape[0]), flags=cv2.INTER_CUBIC)

        # Adjust bounding box coords with padding and clamp within image bounds
        x1 = max(0, box[0] - padding)
        y1 = max(0, box[1] - padding)
        x2 = min(aligned.shape[1], box[2] + padding)
        y2 = min(aligned.shape[0], box[3] + padding)

        cropped = aligned[y1:y2, x1:x2]
        if cropped.size == 0:
            logging.info(f"Empty crop after alignment.")
            return None

        resized = cv2.resize(cropped, target_size)

        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < blur_threshold:
            logging.info(f"Skipped blurry image (Laplacian variance={lap_var:.2f})")
            return None

        return resized

    except Exception as e:
        logging.warning(f"Alignment failed: {e}")
        return None


def process_image(person_folder, img_file):
    input_path = os.path.join(DATASET_ROOT, person_folder, img_file)
    output_path = os.path.join(OUTPUT_ROOT, person_folder, img_file)

    aligned_face = align_and_crop_face(input_path)
    if aligned_face is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_bgr)
    else:
        logging.info(f"No aligned face saved: {input_path}")

def main():
    persons = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    all_tasks = []

    for person in persons:
        img_files = [f for f in os.listdir(os.path.join(DATASET_ROOT, person)) if is_image_file(f)]
        for img_file in img_files:
            all_tasks.append((person, img_file))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_image, person, img_file): (person, img_file) for person, img_file in all_tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                person, img_file = futures[future]
                logging.error(f"Error processing {person}/{img_file}: {e}")

    print("✅ Face cropping and alignment completed.")

if __name__ == '__main__':
    main()
