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
Document-Signature-Detection/
│
├── src/                                        
│   │
│   ├── api/                                    
│   │   ├── __init__.py
│   │   ├── app.py                              # FastAPI app initialization & setup
│   │   └── routers/                            
│   │       ├── __init__.py
│   │       ├── auth.py                         # Login and token generation endpoints
│   │       ├── health.py                       # System and storage status checks
│   │       ├── upload.py                       # Direct image upload endpoints
│   │       ├── files.py                        # MinIO file retrieval and deletion
│   │       └── detect.py                       # Main document & signature ML pipeline
│   │
│   ├── auth/                                   
│   │   ├── __init__.py
│   │   ├── jwt_handler.py                      # JWT encoding and decoding utilities
│   │   ├── oauth2.py                           # Route protection and security scopes
│   │   └── users.py                            # User credentials and database mock
│   │
│   ├── core/                                   
│   │   ├── __init__.py
│   │   ├── config.py                           # Environment variables and settings
│   │   └── dependencies.py                     # ML models and storage client injection
│   │
│   ├── dataset/                                
│   │   ├── configs/
│   │   │   └── dataset.yaml                    # YOLOv8 dataset splits and classes
│   │   └── augmentor.py                        # Image rotation, noise, and adjustments
│   │
│   ├── docker/                                 
│   │   ├── docker-compose.yml                  # Container orchestration (API + MinIO)
│   │   └── Dockerfile                          # Python 3.11 API image definition
│   │
│   ├── docs/                                   
│   │   ├── api_design.md                       # API endpoint specifications
│   │   └── architecture.md                     # System architecture documentation
│   │
│   ├── logs/                                   
│   │   └── app.log                             # Auto-generated application logs
│   │
│   ├── middleware/                             
│   │   ├── __init__.py
│   │   └── request_logger.py                   # Custom HTTP request/response logging
│   │
│   ├── models/                                 
│   │   ├── document_classifier.pt              # Trained document classification weights
│   │   └── signature_yolov8_v2.pt              # Trained YOLOv8 signature detection weights
│   │
│   ├── services/                               
│   │   ├── __init__.py
│   │   ├── detection_service.py                # Core logic for YOLOv8 model execution
│   │   ├── enhancement_service.py              # Logic for triggering image improvements
│   │   ├── storage_service.py                  # Routing files to specific MinIO buckets
│   │   ├── upload_service.py                   # Standard upload processing logic
│   │   └── validation_service.py               # Image dimension and size checks
│   │
│   ├── storage/                                
│   │   ├── __init__.py
│   │   └── minio_client.py                     # MinIO connection and S3 wrapper methods
│   │
│   ├── tests/                                  
│   │   ├── __init__.py
│   │   ├── test_auth.py                        # Unit tests for JWT and login
│   │   ├── test_detect.py                      # Unit tests for the ML pipeline
│   │   ├── test_health.py                      # Unit tests for system health
│   │   ├── test_upload.py                      # Unit tests for MinIO uploads
│   │   └── test_validation.py                  # Unit tests for file formatting
│   │
│   ├── training/                               
│   │   ├── __init__.py
│   │   ├── detector.py                         # YOLOv8 loading and bounding box isolation
│   │   ├── enhancer.py                         # CLAHE, denoising, and blur detection
│   │   ├── pipeline.py                         # Orchestrator tying ML models together
│   │   ├── train.py                            # CLI script for fine-tuning YOLOv8
│   │   └── pdf_utils.py                        # Poppler-based PDF-to-image extraction
│   │
│   └── utils/                                  
│       ├── __init__.py
│       ├── logger.py                           # Loguru structured JSON log configuration
│       └── file_validator.py                   # Magic byte checks for secure uploads
│
├── main.py                                     # Uvicorn server entry point
├── .env                                        # Local environment variables
├── .env.example                                # Template for environment setup
├── .gitignore                                  # Git exclusion rules
├── requirements.txt                            # Python package dependencies
└── README.md                                   # Project documentation
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
