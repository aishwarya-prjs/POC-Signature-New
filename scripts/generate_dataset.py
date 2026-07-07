import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ======================================================
# CONFIGURATION
# ======================================================

SIGNATURE_DATASET = r"C:\Users\ayraw\Downloads\signature_ds_combined"

OUTPUT_DIR = "synthetic_dataset"

NUM_IMAGES = 3000

PAGE_WIDTH = 1024
PAGE_HEIGHT = 1400

MIN_SIGNATURE_WIDTH = 150
MAX_SIGNATURE_WIDTH = 350

random.seed(42)

# ======================================================
# CREATE YOLO DATASET FOLDERS
# ======================================================

train_images = Path(OUTPUT_DIR) / "images" / "train"
val_images = Path(OUTPUT_DIR) / "images" / "val"
test_images = Path(OUTPUT_DIR) / "images" / "test"

train_labels = Path(OUTPUT_DIR) / "labels" / "train"
val_labels = Path(OUTPUT_DIR) / "labels" / "val"
test_labels = Path(OUTPUT_DIR) / "labels" / "test"

for folder in [
    train_images,
    val_images,
    test_images,
    train_labels,
    val_labels,
    test_labels,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ======================================================
# COLLECT SIGNATURE IMAGES
# ======================================================

signature_files = []

for root, dirs, files in os.walk(SIGNATURE_DATASET):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            signature_files.append(os.path.join(root, file))

print("=" * 50)
print("Total signatures found :", len(signature_files))
print("=" * 50)

# ======================================================
# GENERATE DATASET
# ======================================================

print("\nGenerating synthetic dataset...")

for i in tqdm(range(NUM_IMAGES)):

    signature_path = random.choice(signature_files)

    signature = cv2.imread(signature_path)

    if signature is None:
        continue

    # ------------------------------------
    # White A4 page
    # ------------------------------------
    page = np.ones((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) * 255

    h, w = signature.shape[:2]

    target_width = random.randint(
        MIN_SIGNATURE_WIDTH,
        MAX_SIGNATURE_WIDTH,
    )

    scale = target_width / w

    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w >= PAGE_WIDTH or new_h >= PAGE_HEIGHT:
        continue

    signature = cv2.resize(signature, (new_w, new_h))

    # ------------------------------------
    # Random Position
    # ------------------------------------

    x = random.randint(30, PAGE_WIDTH - new_w - 30)
    y = random.randint(30, PAGE_HEIGHT - new_h - 30)

    page[y:y + new_h, x:x + new_w] = signature

    # ------------------------------------
    # Split train / val / test
    # ------------------------------------

    r = random.random()

    if r < 0.8:
        img_folder = train_images
        lbl_folder = train_labels

    elif r < 0.9:
        img_folder = val_images
        lbl_folder = val_labels

    else:
        img_folder = test_images
        lbl_folder = test_labels

    # ------------------------------------
    # Save Image
    # ------------------------------------

    image_name = f"img_{i:05d}.jpg"

    image_path = img_folder / image_name

    cv2.imwrite(str(image_path), page)

    # ------------------------------------
    # YOLO Label
    # ------------------------------------

    x_center = (x + new_w / 2) / PAGE_WIDTH
    y_center = (y + new_h / 2) / PAGE_HEIGHT

    width = new_w / PAGE_WIDTH
    height = new_h / PAGE_HEIGHT

    label_name = f"img_{i:05d}.txt"

    label_path = lbl_folder / label_name

    with open(label_path, "w") as f:
        f.write(
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

print("\nDataset generation completed!")

# ======================================================
# CREATE dataset.yaml
# ======================================================

yaml_text = """path: synthetic_dataset

train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: signature
"""

yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("\nDataset YAML created.")

print("\nDataset Structure")

print("Train Images :", train_images)
print("Validation Images :", val_images)
print("Test Images :", test_images)

print("Train Labels :", train_labels)
print("Validation Labels :", val_labels)
print("Test Labels :", test_labels)

print("\nDone.")