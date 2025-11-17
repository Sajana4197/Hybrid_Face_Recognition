import os
import shutil

base_dir = r"C:\Users\ASUS\Desktop\nFilterd"
output_dir = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_ijbb"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Valid image extensions
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

for root, dirs, files in os.walk(base_dir):
    if root == base_dir:
        continue  # skip base folder itself

    images = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()

    selected = images[:3]  # first 3 images

    for img in selected:
        src_path = os.path.join(root, img)

        # To avoid filename conflicts, prefix with folder name
        person_name = os.path.basename(root)
        new_name = f"{person_name}_{img}"

        dst_path = os.path.join(output_dir, new_name)

        shutil.copy(src_path, dst_path)

    if selected:
        print(f"Copied {len(selected)} images from {root}")
