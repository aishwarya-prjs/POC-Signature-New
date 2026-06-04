"""
Data Augmentation for Signature Detection POC
Expands your 100 manual samples to a larger training set using
signature-specific augmentation (no horizontal flips, careful rotation).
"""

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


class SignatureAugmentor:
    """
    Augments signature images while preserving their visual integrity.
    Generates ~10-15x more data from your 100 manual samples.
    """

    def __init__(self, output_dir: str = "data/augmented", target_per_image: int = 12):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_image = target_per_image

    def augment_dataset(self, source_dir: str) -> int:
        """Augment all images in source_dir and save to output_dir."""
        source = Path(source_dir)
        image_extensions = {".jpg", ".jpeg", ".png"}
        images = [p for p in source.rglob("*") if p.suffix.lower() in image_extensions]

        logger.info(f"Augmenting {len(images)} images → ~{len(images) * self.target_per_image} total")
        total = 0

        for img_path in tqdm(images, desc="Augmenting"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            label_path = img_path.with_suffix(".txt")
            label_content = label_path.read_text() if label_path.exists() else None

            augmented = self._augment_image(img)
            for i, aug_img in enumerate(augmented):
                stem = f"{img_path.stem}_aug{i:03d}"
                out_path = self.output_dir / f"{stem}.jpg"
                cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

                # Copy label unchanged (bounding box coords don't change for most augments)
                if label_content:
                    (self.output_dir / f"{stem}.txt").write_text(label_content)
                total += 1

        # Also copy originals
        for img_path in images:
            out_path = self.output_dir / img_path.name
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(out_path), img)
                label_path = img_path.with_suffix(".txt")
                if label_path.exists():
                    (self.output_dir / label_path.name).write_text(label_path.read_text())
            total += 1

        logger.info(f"Augmentation complete: {total} total images in {self.output_dir}")
        return total

    def _augment_image(self, img: np.ndarray) -> List[np.ndarray]:
        """Apply signature-safe augmentations and return a list of variants."""
        results = []
        h, w = img.shape[:2]

        # 1. Slight rotation (signatures can be slightly tilted, but not flipped)
        for angle in [-8, -4, 4, 8]:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
            results.append(rotated)

        # 2. Brightness / contrast variation (simulate scan quality)
        for alpha, beta in [(0.85, 10), (1.15, -10), (0.7, 20), (1.3, -15)]:
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            results.append(adjusted)

        # 3. Gaussian noise (simulate camera noise / poor scan)
        noisy = img.copy().astype(np.float32)
        noise = np.random.normal(0, 8, noisy.shape)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        results.append(noisy)

        # 4. Slight blur (simulate out-of-focus scan)
        blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
        results.append(blurred)

        # 5. Sharpened (simulate over-processed scan)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        results.append(sharpened)

        # 6. Scale crop (simulate partial view / zoom)
        scale = random.uniform(0.85, 0.95)
        nh, nw = int(h * scale), int(w * scale)
        y1 = random.randint(0, h - nh)
        x1 = random.randint(0, w - nw)
        cropped = img[y1:y1 + nh, x1:x1 + nw]
        cropped = cv2.resize(cropped, (w, h))
        results.append(cropped)

        return results[:self.target_per_image]

    def split_dataset(
        self,
        source_dir: str,
        output_base: str = "data/processed",
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
    ) -> Tuple[int, int, int]:
        """
        Split a directory of images into train/val/test sets.
        Returns (train_count, val_count, test_count).
        """
        source = Path(source_dir)
        base = Path(output_base)

        for split in ["train", "val", "test"]:
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png"}
        images = sorted([p for p in source.rglob("*") if p.suffix.lower() in image_extensions])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        counts = {}
        for split_name, split_images in splits.items():
            for img_path in split_images:
                import shutil
                dest_img = base / "images" / split_name / img_path.name
                shutil.copy2(img_path, dest_img)
                label_path = img_path.with_suffix(".txt")
                if label_path.exists():
                    dest_label = base / "labels" / split_name / label_path.name
                    shutil.copy2(label_path, dest_label)
            counts[split_name] = len(split_images)
            logger.info(f"{split_name}: {len(split_images)} images")

        return counts["train"], counts["val"], counts["test"]
