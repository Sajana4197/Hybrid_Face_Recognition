import os
import shutil

base_dir = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\ijb-testsuite\ijb\IJBB\loose_crop"
output_dir = r"D:\FYP\Hybrid_Face_Recognition\supportive\dataset_test_ijbb"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# List everything in the folder
files = os.listdir(base_dir)

# Take first 50 files
selected = files[:50]

# Copy them
for f in selected:
    src = os.path.join(base_dir, f)
    dst = os.path.join(output_dir, f)
    shutil.copy(src, dst)

print("Copied 50 files successfully!")
