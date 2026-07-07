from ultralytics import YOLO

# Load your existing signature detector
model = YOLO("SRC/models/signature_yolov8_v2.pt")

# Fine-tune on your synthetic dataset
model.train(
    data="synthetic_dataset/dataset.yaml",
    epochs=30,
    imgsz=640,          # Smaller image size for CPU
    batch=2,            # Safe batch size
    workers=0,          # Recommended for Windows + CPU
    device="cpu",       # Force CPU training
    project="runs",
    name="signature_detector_v3",
    pretrained=True,
    patience=10,        # Stop early if no improvement
    save=True
)