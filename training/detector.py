

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — install with: pip install ultralytics")


CLASS_NAMES = {0: "signature", 1: "initials", 2: "stamp"}
CLASS_COLORS = {
    0: (0, 200, 100),   # Green for signature
    1: (100, 150, 255), # Blue for initials
    2: (255, 150, 0),   # Orange for stamp
}


@dataclass
class Detection:
    """A single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "width": x2 - x1,
            "height": y2 - y1,
            "area": self.area,
        }


@dataclass
class InferenceResult:
    """Full result for a single image inference."""
    image_path: str
    detections: List[Detection]
    image_shape: Tuple[int, int]
    inference_time_ms: float
    enhanced: bool = False
    quality_score: Optional[float] = None

    @property
    def has_signature(self) -> bool:
        return any(d.class_id == 0 for d in self.detections)

    @property
    def signatures(self) -> List[Detection]:
        return [d for d in self.detections if d.class_id == 0]

    @property
    def verified(self) -> bool:
        """Verified if at least one signature found above confidence threshold."""
        return any(d.confidence >= 0.6 for d in self.signatures)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_shape": {"height": self.image_shape[0], "width": self.image_shape[1]},
            "inference_time_ms": round(self.inference_time_ms, 2),
            "enhanced": self.enhanced,
            "quality_score": self.quality_score,
            "has_signature": self.has_signature,
            "verified": self.verified,
            "signature_count": len(self.signatures),
            "total_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


class SignatureDetector:
    """
    YOLOv8-based signature detector.
    Supports training, inference, and signature isolation.
    """

    def __init__(
        self,
        model_path: str = "models/signature_yolov8.pt",
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Falls back to yolov8n if custom model not found."""
        if not YOLO_AVAILABLE:
            return

        if self.model_path.exists():
            logger.info(f"Loading custom model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
        else:
            logger.warning(
                f"Custom model not found at {self.model_path}. "
                "Loading yolov8n as base — run train() to create a fine-tuned model."
            )
            self.model = YOLO("yolov8n.pt")

    def train(
        self,
        data_config: str = "configs/dataset.yaml",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/train",
        name: str = "signature_detector",
        resume: bool = False,
    ):
        """
        Fine-tune YOLOv8 on your signature dataset.
        After training, best weights are at runs/train/signature_detector/weights/best.pt
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")

        logger.info(f"Starting training: {epochs} epochs, batch={batch}, imgsz={imgsz}")
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=self.device,
            resume=resume,
            patience=20,
            save_period=10,
            workers=4,
            # Signature-specific settings
            fliplr=0.0,   # Never flip horizontally — breaks signature chirality
            degrees=8.0,  # Allow slight rotation
            translate=0.1,
            scale=0.3,
            mosaic=0.5,
        )
        logger.info(f"Training complete. Results: {results}")

        # Copy best weights to models/
        best_weights = Path(project) / name / "weights" / "best.pt"
        if best_weights.exists():
            self.model_path.parent.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(best_weights, self.model_path)
            logger.info(f"Best weights saved to: {self.model_path}")

        return results

    def predict(
        self,
        image: np.ndarray,
        image_path: str = "",
        enhanced: bool = False,
        quality_score: Optional[float] = None,
    ) -> InferenceResult:
        """Run inference on a single BGR image."""
        import time

        if not YOLO_AVAILABLE or self.model is None:
            raise RuntimeError("Model not available")

        h, w = image.shape[:2]
        start = time.perf_counter()

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = image[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                ))

        return InferenceResult(
            image_path=image_path,
            detections=detections,
            image_shape=(h, w),
            inference_time_ms=elapsed_ms,
            enhanced=enhanced,
            quality_score=quality_score,
        )

    def predict_file(self, image_path: str, **kwargs) -> InferenceResult:
        """Load image from file and run inference."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict(image, image_path=str(image_path), **kwargs)

    def isolate_signatures(
        self,
        image: np.ndarray,
        result: InferenceResult,
        output_dir: str = "outputs/signatures",
        padding: int = 10,
    ) -> List[str]:
        """
        Crop and save each detected signature to output_dir.
        Returns list of saved file paths.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        saved = []
        h, w = image.shape[:2]

        stem = Path(result.image_path).stem if result.image_path else "image"

        for i, det in enumerate(result.signatures):
            x1, y1, x2, y2 = det.bbox
            # Add padding
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

            crop = image[y1:y2, x1:x2]
            out_path = output / f"{stem}_sig{i:02d}_conf{det.confidence:.2f}.png"
            cv2.imwrite(str(out_path), crop)
            saved.append(str(out_path))
            logger.info(f"Isolated signature saved: {out_path}")

        return saved

    def draw_results(self, image: np.ndarray, result: InferenceResult) -> np.ndarray:
        """Draw bounding boxes and labels on image. Returns annotated copy."""
        annotated = image.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_id, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        status = "VERIFIED ✓" if result.verified else ("DETECTED" if result.has_signature else "NOT FOUND")
        status_color = (0, 200, 100) if result.verified else (0, 120, 255) if result.has_signature else (0, 0, 220)
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        return annotated
