# Signature Detection POC

YOLOv8-based signature detection and document verification system.
Detects signatures from cheques, forms, and agreements using two ML models.

---

## What It Does

- Accepts a document image (cheque / form / agreement)
- Classifies the document type automatically
- Detects and isolates the signature region
- Stores everything in MinIO object storage
- Returns a structured JSON result

---
## Project Structure

```
POC-Signature/
├── api/                          
│   ├── routers/
│   │   ├── health.py             # GET /health, GET /storage/health
│   │   ├── upload.py             # POST /upload
│   │   ├── files.py              # GET /files, GET /files/{name}, DELETE /files/{name}
│   │   └── detect.py             # POST /detect — classify document + find signature
│   ├── src/
│   │   └── storage/
│   │       └── minio_client.py   # MinIO connect, upload, download, presigned URLs
│   ├── models/                   # Trained .pt files stored here
│   │   ├── document_classifier.pt
│   │   └── signature_yolov8_v2.pt
│   ├── dependencies.py           # Loads both models + storage on startup
│   ├── main.py                   # Registers all routers, starts FastAPI app
│   ├── .env.example              # Environment variable template
│   └── requirements.txt          # fastapi, uvicorn, minio, ultralytics, opencv
│
├── docker/                       
│   ├── docker-compose.yml        # Spins up MinIO + API as Docker services
│   ├── Dockerfile                # Builds the API container
│   └── .env.example              # MinIO credentials template for Docker
│
├── dataset/                      
│   ├── configs/
│   │   └── dataset.yaml          # YOLOv8 dataset config — paths, classes, splits
│   └── augmentor.py              # Augments images — rotation, brightness, noise
│
├── training/                     
│   ├── detector.py               # YOLOv8 model wrapper — train, predict, isolate
│   ├── enhancer.py               # Quality scoring — blur, contrast, CLAHE, binarize
│   ├── pipeline.py               # Full pipeline — assess, enhance, detect, store
│   └── train.py                  # Training script — runs on Google Colab T4 GPU
│
├── .gitignore
└── README.md
```
---

## Model Files

The `.pt` model files are not included in this repo due to size.

Download from Google Drive: `[share your drive link here]`

Place them in:
api/models/document_classifier.pt
api/models/signature_yolov8_v2.pt

### What each model does

| Model | File | Purpose |
|---|---|---|
| Document Classifier | `document_classifier.pt` | Identifies cheque / form / agreement / others |
| Signature Detector | `signature_yolov8_v2.pt` | Finds and crops the signature region |

---

## Team Contributions

| Member | Repo | Responsible For |
|---|---|---|
| Vijay | POC-Signature | API, Docker, MinIO storage |
| Aiswarya | POC-Signature-Training | Dataset, Training, Model weights |

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/2108vijay/POC-Signature.git
cd POC-Signature
```

### 2. Set up environment
```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Start MinIO via Docker
```bash
cd docker
docker-compose up
```

MinIO Console → `http://localhost:9001`

### 4. Download model files
Download both `.pt` files from Google Drive and place in `api/models/`

### 5. Start the API
```bash
cd api
python3 main.py
```

API docs → `http://localhost:8000/docs`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | API + MinIO status |
| GET | `/storage/health` | MinIO bucket details |
| POST | `/upload` | Upload image to MinIO |
| POST | `/detect` | Classify document + detect signature |
| GET | `/files` | List all stored files |
| GET | `/files/{name}` | Get presigned URL for a file |
| DELETE | `/files/{name}` | Delete a file |
| GET | `/docs` | Swagger UI |

---

## Sample JSON Output

Upload a cheque/form/agreement to `/detect` and get back:

```json
{
  "document": "cheque.jpg",
  "document_type": "cheque",
  "document_confidence": 0.94,
  "signature_found": true,
  "signature_verified": true,
  "total_signatures": 1,
  "detections": [
    {
      "signature_id": 1,
      "confidence": 0.89,
      "verified": true,
      "bounding_box": {
        "x1": 420,
        "y1": 310,
        "x2": 680,
        "y2": 410,
        "width": 260,
        "height": 100
      },
      "crop_url": "http://localhost:9000/..."
    }
  ],
  "stored_as": "uploads/abc123_cheque.jpg"
}
```

---

## Document Types

| Label | Description |
|---|---|
| `cheque` | Bank cheque with MICR line |
| `form` | Filled application or government form |
| `agreement` | Legal document or stamp paper |
| `others` | Unknown document type |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Image Enhancement | OpenCV, CLAHE |
| Storage | MinIO |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Training | Google Colab (T4 GPU) |
