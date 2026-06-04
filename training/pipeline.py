

import json
import time
from pathlib import Path
from typing import Optional
from src.enhancement.enhancer import assess_quality, enhance, process_image, QualityReport
import cv2
import numpy as np
from loguru import logger


class SignaturePipeline:
    """
    Full pipeline: image in → verified result + stored artefacts out.
    """

    def __init__(
        self,
        model_path: str = "models/signature_yolov8.pt",
        confidence: float = 0.5,
        output_dir: str = "outputs",
        minio_endpoint: str = None,
        enable_storage: bool = True,
        device: str = "cpu",
    ):
        from src.detection.detector import SignatureDetector
        from src.storage.minio_client import MinioStorageClient

        # No more assessor/enhancer initialization needed here!

        self.detector = SignatureDetector(
            model_path=model_path,
            confidence=confidence,
            device=device,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "signatures").mkdir(exist_ok=True)
        (self.output_dir / "annotated").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        self.storage = None
        if enable_storage:
            try:
                self.storage = MinioStorageClient()
                if self.storage.is_connected:
                    logger.info("MinIO storage enabled")
                else:
                    self.storage = None
                    logger.warning("MinIO not reachable — running without cloud storage")
            except Exception as e:
                logger.warning(f"MinIO disabled: {e}")

    def run(self, image_path: str, save_local: bool = True) -> dict:
        """
        Process one image through the full pipeline.
        Returns a complete result dict.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # 1. Quality assessment
        quality = self.assessor.assess(image, str(image_path))
        logger.info(f"Quality: {quality.summary()}")

        # 2. Enhancement if needed
        enhanced_image = image
        was_enhanced = False
        if quality.needs_enhancement:
            enhanced_image = self.enhancer.enhance(image, quality)
            was_enhanced = True
            logger.info("Image enhanced")

        # 3. Detection
        result = self.detector.predict(
            enhanced_image,
            image_path=str(image_path),
            enhanced=was_enhanced,
            quality_score=quality.overall_score,
        )
        logger.info(
            f"Detection: {len(result.signatures)} signature(s) found, "
            f"verified={result.verified}, {result.inference_time_ms:.1f}ms"
        )

        # 4. Isolate signatures
        signature_crops = []
        if result.has_signature and save_local:
            signature_crops = self.detector.isolate_signatures(
                enhanced_image, result,
                output_dir=str(self.output_dir / "signatures"),
            )

        # 5. Save annotated image
        annotated_path = None
        if save_local:
            annotated = self.detector.draw_results(enhanced_image, result)
            annotated_path = self.output_dir / "annotated" / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated)

        # 6. Build report
        total_ms = (time.perf_counter() - start) * 1000
        report = {
            "image": str(image_path),
            "quality": {
               "grade": "A" if quality.overall_score >= 80 else "B" if not quality.needs_enhancement else "C",
                "score": quality.overall_score, # Already rounded in enhancer.py
                "enhanced": was_enhanced,
                "blur": quality.blur_score,
            },
            "detection": result.to_dict(),
            "output": {
                "annotated": str(annotated_path) if annotated_path else None,
                "signature_crops": signature_crops,
            },
            "total_ms": round(total_ms, 2),
        }

        # 7. Save report locally
        if save_local:
            report_path = self.output_dir / "reports" / f"{image_path.stem}_report.json"
            report_path.write_text(json.dumps(report, indent=2))

        # 8. Store to MinIO
        if self.storage and self.storage.is_connected:
            try:
                stored = self.storage.store_detection_result(
                    enhanced_image, report, signature_crops
                )
                report["minio"] = stored
            except Exception as e:
                logger.warning(f"MinIO storage failed: {e}")

        return report

    def run_batch(self, input_dir: str, extensions=(".jpg", ".jpeg", ".png")) -> list:
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        images = [p for p in input_dir.rglob("*") if p.suffix.lower() in set(extensions)]
        logger.info(f"Batch processing {len(images)} images from {input_dir}")

        results = []
        for img_path in images:
            try:
                result = self.run(str(img_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed {img_path.name}: {e}")
                results.append({"image": str(img_path), "error": str(e)})

        # Summary
        verified = sum(1 for r in results if r.get("detection", {}).get("verified", False))
        detected = sum(1 for r in results if r.get("detection", {}).get("has_signature", False))
        logger.info(
            f"Batch complete: {len(images)} images | "
            f"{detected} with signatures | {verified} verified"
        )
        return results
